from rail.creation.engine import Creator
from rail.core.stage import RailStage
from rail.core.data import Hdf5Handle
from ceci.config import StageParameter as Param
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15, w0waCDM
from scipy import interpolate


class FSPSPhotometryCreator(Creator):
    """
    Derived class of Creator that generate synthetic photometric data from the rest-frame SED model
    generated with the FSPSSedModeler class.
    The user is required to provide galaxy redshifts and filter information in an .npy format for the code to run.
    The restframe SEDs are stored in a pickle file or passed as ModelHandle.
    Details of what each file should contain are explicited in config_options.
    The output is a Fits table containing magnitudes.
    """

    name = "FSPS_Photometry_Creator"

    config_options = RailStage.config_options.copy()
    config_options.update(filter_data=Param(str, 'lsst_filters.npy', msg='npy file containing the structured numpy '
                                                                         'array of the survey filter wavelengths and'
                                                                         ' transmissions'),
                          rest_frame_wavelengths_key = Param(str, 'wavelength', msg='Rest-frame wavelengths dataset'
                                                                                    'keyword name'),
                          rest_frame_sed_models=Param(str, 'restframe_seds.hdf5',
                                                      msg='Rest-frame seds hdf5 filename'),
                          rest_frame_sed_models_key=Param(str, 'restframe_seds',
                                                      msg='Rest-frame seds dataset keyword name'),
                          redshifts_key=Param(str, 'redshifts', msg='Redshifts dataset keyword name'),
                          Om0=Param(float, 0.3, msg='Omega matter at current time'),
                          Ode0=Param(float, 0.7, msg='Omega dark energy at current time'),
                          w0=Param(float, -1, msg='Dark energy equation-of-state parameter at current time'),
                          wa=Param(float, 0., msg='Slope dark energy equation-of-state evolution with scale factor'),
                          h=Param(float, 0.7, msg='Dimensionless hubble constant'),
                          use_planck_cosmology=Param(bool, False, msg='True to overwrite the cosmological parameters'
                                                                      'to their Planck2015 values'),
                          physical_units=Param(bool, False), msg='False (True) for rest-frame spectra in units of'
                                                                 'Lsun/Hz (erg/s/Hz)')

    inputs = [("model", Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize class.
        The _b and _c tuples for jax are composed of None or 0, depending on whether you don't or do want the
        array axis to map over for all arguments.
        Parameters
        ----------
        args:
        comm:
        """
        RailStage.__init__(self, args, comm=comm)

        if self.config.use_planck_cosmology:
            self.cosmology = Planck15
        else:
            self.cosmology = w0waCDM(self.config.h * 100, self.config.Om0, self.config.Ode0,
                                     w0=self.config.w0, wa=self.config.wa)

        if (self.config.Om0 < 0.) | (self.config.Om0 > 1.):
            raise ValueError("The mass density at the current time {self.config.Om0} is outside of allowed"
                             " range 0. < Om0 < 1.")
        if (self.config.Ode0 < 0.) | (self.config.Ode0 > 1.):
            raise ValueError("The dark energy density at the current time {self.config.Ode0} is outside of allowed"
                             " range 0. < Ode0 < 1.")
        if (self.config.h < 0.) | (self.config.h > 1.):
            raise ValueError("The dimensionless Hubble constant {self.config.h} is outside of allowed"
                             " range 0 < h < 1")

        self.filter_data = np.load(self.config.filter_data)
        self.filter_names = np.array([key for key in self.filter_data.dtype.fields
                                      if 'wave' in key])
        self.filter_wavelengths = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
                                            if 'wave' in key])
        self.filter_transmissions = np.array([self.filter_data[key] for key in self.filter_data.dtype.fields
                                              if 'trans' in key])

        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Creator

        Keywords
        --------
        model : `str`
            Either a path pointing to a file that can be read to obtain the trained model,
            or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """

        if isinstance(self.config.rest_frame_sed_models, str):  # pragma: no cover
            self.model = self.set_data("model", data=None, path=self.config.rest_frame_sed_models)
            self.config["model"] = self.config.rest_frame_sed_models
            return self.model

        if isinstance(self.config.rest_frame_sed_models, Hdf5Handle):  # pragma: no cover
            if self.config.rest_frame_sed_models.has_path:
                self.config["model"] = self.config.rest_frame_sed_models.path
                self.model = self.set_data("model", self.config.rest_frame_sed_models)
            return self.model

    def _get_apparent_magnitudes(self):
        """

        Returns
        -------
        apparent_magnitudes: numpy.array
            Array of shape (n_galaxies, n_bands) containing the computed apparent AB magnitudes

        """

        apparent_magnitudes = {}

        for i in self.split_tasks_by_rank(range(len(self.model[self.config.rest_frame_sed_models_key]))):

            if self.config.physical_units:
                restframe_sed = self.model[self.config.rest_frame_sed_models_key][i]
            else:
                solar_luminosity_erg_s = 3.826 * 10**33
                restframe_sed = self.model[self.config.rest_frame_sed_models_key][i] * solar_luminosity_erg_s

            Mpc_in_cm = 3.08567758128 * 10 ** 24
            speed_of_light_cm_s = 2.9979245800 * 10 ** 18
            lum_dist_cm = self.cosmology.luminosity_distance(self.model[self.config.redshifts_key][i]).value * Mpc_in_cm

            observedframe_sed_erg_s_cm2_Hz = (1 + self.model[self.config.redshifts_key][i]) ** 2 * restframe_sed / \
                (4 * np.pi * (1 + self.model[self.config.redshifts_key][i]) * lum_dist_cm ** 2)

            observedframe_wavelength = self.model[self.config.rest_frame_wavelengths_key] * \
                (1 + self.model[self.config.redshifts_key][i])
            observedframe_wavelength_in_Hz = 2.9979245800 * 10 ** 18 / observedframe_wavelength

            magnitudes = []

            for j in range(len(self.filter_transmissions)):
                filter_wavelength_in_hz = speed_of_light_cm_s / self.filter_wavelengths[j, :]
                interp_function = interpolate.interp1d(filter_wavelength_in_hz, self.filter_transmissions[j, :],
                                                       bounds_error=False, fill_value=0)
                filt_interp = interp_function(observedframe_wavelength_in_Hz)
                numerator = np.trapz(observedframe_sed_erg_s_cm2_Hz * filt_interp / observedframe_wavelength,
                                     x=observedframe_wavelength)
                denominator = np.trapz(filt_interp / observedframe_wavelength,
                                       x=observedframe_wavelength)
                mag_ab = -2.5 * np.log10(numerator / denominator) - 48.6
                magnitudes.append(mag_ab)

            apparent_magnitudes[i] = magnitudes

        if self.comm is not None:  # pragma: no cover
            apparent_magnitudes = self.comm.gather(apparent_magnitudes)

            if self.rank != 0:  # pragma: no cover
                return None, None

            apparent_magnitudes = {k: v for a in apparent_magnitudes for k, v in a.items()}

        apparent_magnitudes = np.array([apparent_magnitudes[i]
                                        for i in range(len(self.model[self.config.rest_frame_sed_models_key]))])

        return apparent_magnitudes

    def sample(self, **kwargs):
        r"""
        Creates observed and absolute magnitudes for population of galaxies and stores them into Fits table.

        This is a method for running in interactive mode.
        In pipeline mode, the subclass `run` method will be called by itself.

        Parameters
        ----------

        Returns
        -------
        output: astropy.table.Table
            Fits table storing galaxy magnitudes and redshifts.

        Notes
        -----
        This method calls the `run` method.
        Finally, the `FitsHandle` associated to the `output` tag is returned.

        """

        self.config.update(**kwargs)
        self.run()
        self.finalize()
        output = self.get_handle("output")
        return output

    def run(self):
        """
        This function computes apparent AB magnitudes in the provided wavebands for all the galaxies
        in the population having rest-frame SEDs computed by FSPS.
        It then stores magnitudes, redshifts and the galaxy indices into an astropy.table.Table.

        Returns
        -------

        """

        apparent_magnitudes = self._get_apparent_magnitudes()

        idxs = np.arange(1, self.model[self.config.redshifts_key] + 1, 1, dtype=int)

        if self.rank == 0:
            output_values = {'id': idxs, 'z': self.model[self.config.redshifts_key], 'app_mags': apparent_magnitudes}
            self.add_data('output', output_values)
