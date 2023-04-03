from rail.creation.engine import Modeler
from rail.core.stage import RailStage
from rail.core.data import ModelHandle, Hdf5Handle
from ceci.config import StageParameter as Param
import fsps
import numpy as np


class FSPSSedModeler(Modeler):
    r"""
    Derived class of Modeler for creating a single galaxy rest-frame SED model using FSPS (Conroy08).

    Notes
    -----
    Only the most important parameters are provided via config_options. The remaining ones from FSPS can be
    provided when creating the rest-frame SED model.

    Install FSPS with the following commands:
    git clone --recursive https://github.com/dfm/python-fsps.git
    cd python-fsps
    python -m pip install .
    """

    name = "FSPS_sed_model"
    config_options = RailStage.config_options.copy()
    config_options.update(zcontinuous=Param(int, 1, msg='Flag for interpolation in metallicity of SSP before CSP'),
                          add_agb_dust_model=Param(bool, True,
                                                   msg='Turn on/off adding AGB circumstellar dust contribution to SED'),
                          add_dust_emission=Param(bool, True,
                                                  msg='Turn on/off adding dust emission contribution to SED'),
                          add_igm_absorption=Param(bool, False,
                                                   msg='Turn on/off adding IGM absorption contribution to SED'),
                          add_neb_emission=Param(bool, False, msg='Turn on/off nebular emission model based on Cloudy'),
                          add_neb_continuum=Param(bool, False, msg='Turn on/off nebular continuum component'),
                          add_stellar_remnants=Param(bool, True, msg='Turn on/off adding stellar remnants contribution '
                                                                     'to stellar mass'),
                          nebemlineinspec=Param(bool, False, msg='True to include emission line fluxes in spectrum'),
                          smooth_velocity=Param(bool, True, msg='True/False for smoothing in '
                                                                'velocity/wavelength space'),
                          smooth_lsf=Param(bool, False, msg='True/False for smoothing SSPs by a wavelength dependent '
                                                            'line spread function'),
                          imf_type=Param(int, 1, msg='IMF type, see FSPS manual, default Chabrier IMF'),
                          min_wavelength=Param(float, 3000, msg='minimum rest-frame wavelength'),
                          max_wavelength=Param(float, 10000, msg='maximum rest-frame wavelength'),
                          sfh_type=Param(int, 0, msg='star-formation history type, see FSPS manual, default SSP'),
                          dust_type=Param(int, 2, msg='attenuation curve for dust type, see FSPS manual, '
                                                      'default Calzetti'),
                          tabulated_sfh_key = Param(str, 'tabulated_sfh', msg='tabulated SFH dataset keyword name'),
                          redshifts_key = Param(str, 'redshift', msg='Redshift dataset keyword name'),
                          stellar_metallicities_key=Param(str, 'stellar_metallicity',
                                                     msg='galaxy stellar metallicities (log10(Z / Zsun)) '
                                                         'dataset keyword name'),
                          stellar_ages_key=Param(str, 'stellar_age', msg='galaxy stellar ages (Gyr) '
                                                                         'dataset keyword name'),
                          velocity_dispersions_key=Param(str, 'stellar_velocity_dispersion',
                                                         msg='stellar velocity dispersions (km/s) '
                                                             'dataset keyword name'),
                          gas_ionizations_key=Param(str, 'gas_ionization',
                                                          msg='gas ionization values dataset keyword name'),
                          gas_metallicities_key=Param(str, 'gas_metallicity',
                                                      msg='gas metallicities (log10(Zgas / Zsun)) dataset '
                                                          'keyword name'),
                          dust_birth_cloud_key=Param(str, 'dust1_birth_cloud',
                                                        msg='dust parameter describing young stellar light attenuation '
                                                            '(dust1 in FSPS) dataset keyword name'),
                          dust_diffuse_key=Param(str, 'dust2_diffuse', msg='dust parameters describing old stellar '
                                                                              'light attenuation (dust2 in FSPS) '
                                                                              'dataset keyword name'),
                          dust_powerlaw_modifier_key=Param(str, 'dust_calzetti_modifier',
                                                              msg='power-law modifiers to the shape of the '
                                                                  'Calzetti et al. (2000) attenuation curve '
                                                                  'dataset keyword name'),
                          dust_emission_gamma_key=Param(str, 'dust_gamma',
                                                           msg='Relative contributions of dust heated at Umin, '
                                                               'parameter of Draine and Li (2007) dust emission model'
                                                               'dataset keyword name'),
                          dust_emission_umin_key=Param(str, 'dust_umin',
                                                          msg='Minimum radiation field strengths, parameter of '
                                                              'Draine and Li (2007) dust emission model, '
                                                              'dataset keyword name'),
                          dust_emission_qpah_key=Param(str, 'dust_qpah',
                                                          msg='Grain size distributions in mass in PAHs, '
                                                              'parameter of Draine and Li (2007) dust emission model,'
                                                              'dataset keyword name'),
                          fraction_agn_bol_lum_key=Param(str, 'f_agn',
                                                            msg='Fractional contributions of AGN wrt stellar bolometric'
                                                                ' luminosity, dataset keyword name'),
                          agn_torus_opt_depth_key=Param(str, 'tau_agn', msg='Optical depths of the AGN dust torii'
                                                                               ' dataset keyword name'),
                          physical_units=Param(bool, False), msg='False (True) for rest-frame spectra in units of'
                                                                 'Lsun/Hz (erg/s/Hz)')

    inputs = [("input", Hdf5Handle)]
    # outputs = [("model", ModelHandle)]
    outputs = [("model", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """
        Initialize Modeler

        Parameters
        ----------
        args:
        comm:

        """
        RailStage.__init__(self, args, comm=comm)

        if self.config.min_wavelength < 0:
            raise ValueError("min_wavelength must be positive, not {self.config.min_wavelength}")
        if (self.config.max_wavelength < 0) | (self.config.max_wavelength <= self.config.min_wavelength):
            raise ValueError("max_wavelength must be positive and greater than min_wavelength,"
                             " not {self.config.max_wavelength}")

        if self.config.zcontinuous not in [0, 1, 2, 3]:
            raise ValueError("zcontinous={} is not valid, allowed values are 0,1,2,3".format(self.config.zcontinuous))

        if self.config.imf_type not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("imf_type={} is not valid, allowed values are 0,1,2,3,4,5".format(self.config.imf_type))

        if self.config.sfh_type not in [0, 1, 3, 4, 5]:
            raise ValueError("sfh_type={} is not valid, allowed values are 0,1,2,3,4,5".format(self.config.sfh_type))

        if self.config.dust_type not in [0, 1, 2, 3, 4, 5, 6]:
            raise ValueError("dust_type={} is not valid, allowed values are 0,1,2,3,4,5,6"
                             .format(self.config.dust_type))

    def _get_rest_frame_seds(self, compute_vega_mags=False, vactoair_flag=False, zcontinuous=0, add_agb_dust_model=True,
                             add_dust_emission=True, add_igm_absorption=False, add_neb_emission=False,
                             add_neb_continuum=True, add_stellar_remnants=True, redshift_colors=False,
                             compute_light_ages=False, nebemlineinspec=True, smooth_velocity=True,
                             smooth_lsf=False, cloudy_dust=False, agb_dust=1.0, tpagb_norm_type=2, dell=0.0,
                             delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0, sbss=0.0, pagb=1.0, zred=0.0,
                             zmet=1, logzsol=0.0, pmetals=2.0, imf_type=2, imf_upper_limit=120, imf_lower_limit=0.08,
                             imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, masscut=150.0,
                             sigma_smooth=0.0, min_wave_smooth=1e3, max_wave_smooth=1e4, gas_logu=-2, gas_logz=0.0,
                             igm_factor=1.0, sfh=0, tau=1.0, const=0.0, sf_start=0.0, sf_trunc=0.0, tage=0.0,
                             fburst=0.0, tburst=11.0, sf_slope=0.0, dust_type=0, dust_tesc=7.0, dust1=0.0, dust2=0.0,
                             dust_clumps=-99., frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, dust1_index=-1.0,
                             mwr=3.1, uvb=1.0, wgp1=1, wgp2=1, wgp3=1, duste_gamma=0.01, duste_umin=1.0,
                             duste_qpah=3.5, fagn=0.0, agn_tau=10.0, tabulated_sfh_files=None,
                             tabulated_lsf_files=None):
        """

        Parameters
        ----------
        compute_vega_mags: bool
            Default to False for AB magnitudes, True for Vega magnitudes
        vactoair_flag: bool
            Defaul to False for vacuum wavelengths, True for air wavelengths
        compute_light_ages: bool
            Default to False, computing mass-weighted (False) or light-weighted (True) ages
        cloudy_dust: bool
            Default to False, including dust in Cloudy tables
        agb_dust: float
            Default to 1., Scales the circumstellar AGB dust emission
        tpagb_norm_type: int
            Default to 2 (Villaume, Conroy, Johnson 2015 normalization), TP-AGB normalization scheme
        dell: float
            Default to 0.0, Shift in log10 bolometric luminosity of the TP-AGB isochrones
        delt: float
            Default to 0.0, Shift in log10 effective temperature of the TP-AGB isochrones
        redgb: float
            Default to 1.0, Modify weight given to RGB, only available with BaSTI isochrones
        agb: float
            Default to 1.0, Modify weight given to TP-AGB
        fcstar: float
            Default to 1.0, Fraction of stars that the Padova isochrones identify as Carbon stars
        fbhb: float
            Default to 0.0, Fraction of horizontal branch stars that are blue, see Conroy+09
        sbss: float
            Default to 0.0, Frequency of blue stragglers, see Conroy+99
        pagb: float
            Default to 1.0, Weight given to the post–AGB phase
        zmet: int
            Default to 1, The metallicity is specified as an integer ranging between 1 and nz.
            If zcontinuous > 0 then this parameter is ignored.
        pmetals: float
            Default to 2.0, The power for the metallicty distribution function
        imf_upper_limit: float
            Default to 120, upper limit of the IMF in solar masses
        imf_lower_limit: float
            Default to 0.08, lower limit of the IMF in solar masses
        imf1: float
            Default to 1.3, logarithmic slope of the IMF over the range 0.08 < M < 0.5 Msun
        imf2: float
            Default to 2.3, logarithmic slope of the IMF over the range 0.5 < M < 1.0 Msun
        imf3: float
            Default to 2.3, logarithmic slope of the IMF over the range 1.0 < M < upper_limit Msun
        vdmc: float
            Default to 0.08, IMF parameter defined in van Dokkum08
        mdave: float
            Default to 0.5, IMF parameter defined in Dave08
        evtype: int
            Default to -1 (all phases), compute SSPs for only the given evolutionary type
        use_wr_spectra: int
            Default to 1, turning on (1) or off(0) the Wolf-Rayet spectral library
        logt_wmb_hot: float
            Default to 0.0, Use the Eldridge (2017) WMBasic hot star library above this value of effective temperature
            or 25,000K, whichever is larger
        masscut: float
            Default to 150.0, truncate the IMF above this value
        igm_factor: float
            Default to 1.0, factor used to scale the IGM optical depth
        tau: numpy.array
            Default to 1.0, array of e-folding times for the SFHs in Gyr, only used if sfh_type in {1,4}
        const: numpy.array
            Default to 0.0, array of mass fractions formed in a constant mode of SF, only used if sfh_type in {1,4}
        sf_start: numpy.array
            Default to 0.0, array of start times of the SFH, only used if sfh_type in {1,4,5}
        sf_trunc: numpy.array
            Default to 0.0, array of truncation times of the SFH, only used if sfh_type in {1,4,5}
        fburst: numpy.array
            Default to 0.0, array of fraction of masses formed in an instantaneous burst of star-formation, only used if
            sfh_type in {1,4}
        tburst: numpy.array
            Default to 11.0, array of ages of the Universe when bursts occur, only used if sfh_type in {1,4}
        sf_slope: numpy.array
            Default to 0.0, array of slopes of the SFR after time sf_trunc, only used if sfh_type in {5}
        dust_tesc: float
            Default to 7.0, stars younger than this value are attenuated both by dust1 and dust2, in units of log(yrs)
        dust_clumps: float
            Default to -99, Dust parameter describing the dispersion of a Gaussian PDF density
            distribution for the old dust
        frac_nodust: float
            Default to 0.0, Fraction of starlight that is not attenuated by the diffuse dust component
        frac_obrun: float
            Default to 0.0, Fraction of the young stars that are not attenuated by dust1 and that do not
            contribute to any nebular emission
        dust_index: float
            Default to -0.7, Power law index of the attenuation curve, only used if dust_type in {0}
        mwr: float
            Default to 3.1, The ratio of total to selective absorption which characterizes the MW extinction curve,
            only used if dust_type in {1}
        uvb: float
            Default to 1.0, Parameter characterizing the strength of the 2175A extinction feature with
            respect to the standard Cardelli et al. determination for the MW, only used if dust_type in {1}
        wgp1: int
            Default to 1, Integer specifying the optical depth in the Witt & Gordon (2000) models
        wgp2: int
            Default to 1, Integer specifying the type of large-scale geometry and extinction curve
        wgp3: int
            Default to 1, Integer specifying the local geometry for the Witt & Gordon (2000) dust models
        physical_units: bool
            Default to False, True (False) for SED units in erg/s/Hz (Lsun/Hz)
        zred: str
            Default to None, If this value is not None, redshifts are read from an external numpy array and if
            redshift_colors=1, the magnitudes will be computed for the spectrum placed at redshift zred[i].
        tabulated_sfh_file: str
            Default to None, path to the file storing the galaxy star-formation history
        tabulated_lsf_file: str
            Default to None, path to the file storing the values for the line-spread function smoothing
        Returns
        -------

        """

        if np.isscalar(tau):
            tau = np.full(len(tage), tau)
            const = np.full(len(tage), const)
            sf_start = np.full(len(tage), sf_start)
            sf_trunc = np.full(len(tage), sf_trunc)
            fburst = np.full(len(tage), fburst)
            tburst = np.full(len(tage), tburst)
            sf_slope = np.full(len(tage), sf_slope)

        if np.isscalar(dust1):
            dust1 = np.full(len(tage), dust1)
        if np.isscalar(dust2):
            dust2 = np.full(len(tage), dust2)
        if np.isscalar(dust1_index):
            dust1_index = np.full(len(tage), dust1_index)

        if np.isscalar(duste_gamma):
            duste_gamma = np.full(len(tage), duste_gamma)
        if np.isscalar(duste_umin):
            duste_umin = np.full(len(tage), duste_umin)
        if np.isscalar(duste_qpah):
            duste_qpah = np.full(len(tage), duste_qpah)

        restframe_wavelengths = {}
        restframe_seds = {}

        for i in self.split_tasks_by_rank(range(len(tage))):
            sp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                        vactoair_flag=vactoair_flag,
                                        zcontinuous=self.config.zcontinuous,
                                        add_agb_dust_model=self.config.add_agb_dust_model,
                                        add_dust_emission=self.config.add_dust_emission,
                                        add_igm_absorption=self.config.add_igm_absorption,
                                        add_neb_emission=self.config.add_neb_emission,
                                        add_neb_continuum=self.config.add_neb_continuum,
                                        add_stellar_remnants=self.config.add_stellar_remnants,
                                        compute_light_ages=compute_light_ages,
                                        nebemlineinspec=self.config.nebemlineinspec,
                                        smooth_velocity=self.config.smooth_velocity,
                                        smooth_lsf=self.config.smooth_lsf,
                                        cloudy_dust=cloudy_dust, agb_dust=agb_dust,
                                        tpagb_norm_type=tpagb_norm_type, dell=dell, delt=delt,
                                        redgb=redgb, agb=agb, fcstar=fcstar, fbhb=fbhb,
                                        sbss=sbss, pagb=pagb, zred=zred[i],
                                        zmet=zmet[i], logzsol=logzsol[i],
                                        pmetals=pmetals, imf_type=self.config.imf_type,
                                        imf_upper_limit=imf_upper_limit,
                                        imf_lower_limit=imf_lower_limit, imf1=imf1, imf2=imf2,
                                        imf3=imf3, vdmc=vdmc, mdave=mdave, evtype=evtype,
                                        masscut=masscut, sigma_smooth=sigma_smooth[i],
                                        min_wave_smooth=self.config.min_wavelength,
                                        max_wave_smooth=self.config.max_wavelength,
                                        gas_logu=gas_logu[i], gas_logz=gas_logz[i],
                                        igm_factor=igm_factor, sfh=self.config.sfh_type,
                                        tau=tau[i], const=const[i], sf_start=sf_start[i],
                                        sf_trunc=sf_trunc[i], tage=tage[i],
                                        fburst=fburst[i], tburst=tburst[i],
                                        sf_slope=sf_slope[i], dust_type=self.config.dust_type,
                                        dust_tesc=dust_tesc, dust1=dust1[i], dust2=dust2[i], dust_clumps=dust_clumps,
                                        frac_nodust=frac_nodust, frac_obrun=frac_obrun,
                                        dust_index=dust_index, dust1_index=dust1_index[i],
                                        mwr=mwr, uvb=uvb, wgp1=wgp1, wgp2=wgp2, wgp3=wgp3,
                                        duste_gamma=duste_gamma[i], duste_umin=duste_umin[i],
                                        duste_qpah=duste_qpah[i],
                                        fagn=frac_bol_lum_agn[i], agn_tau=agn_torus_opt_depths[i])

            if self.config.sfh_type == 3:

                if self.config.zcontinuous == 3:
                    #age_array, sfr_array, metal_array = np.loadtxt(tabulated_sfh_files[i], usecols=(0, 1, 2),
                    #                                               unpack=True)
                    age_array, sfr_array, metal_array = tabulated_sfh_files[i]
                    sp.set_tabular_sfh(age_array, sfr_array, Z=metal_array)
                elif self.config.zcontinuous == 1:
                    #age_array, sfr_array = np.loadtxt(tabulated_sfh_file[i], usecols=(0, 1), unpack=True)
                    age_array, sfr_array = tabulated_sfh_files[i]
                    sp.set_tabular_sfh(age_array, sfr_array, Z=None)
                else:
                    raise ValueError

            if self.config.smooth_lsf:
                assert self.config.smooth_velocity is True, 'lsf smoothing only works if smooth_velocity is True'
                # lsf_values = np.loadtxt(tabulated_lsf_file, usecols=(0, 1))
                # wave = lsf_values[:, 0]  # pragma: no cover
                # sigma = lsf_values[:, 1]  # pragma: no cover
                wave, sigma = tabulated_lsf_files[i]
                sp.set_lsf(wave, sigma, wmin=self.config.min_wavelength,
                           wmax=self.config.max_wavelength)  # pragma: no cover

            restframe_wavelength, restframe_sed_Lsun_Hz = sp.get_spectrum(tage=tage[i], peraa=False)

            selected_wave_range = np.where((restframe_wavelength >= self.config.min_wavelength) &
                                           (restframe_wavelength <= self.config.max_wavelength))
            restframe_wavelength = restframe_wavelength[selected_wave_range]
            restframe_wavelengths[i] = restframe_wavelength

            if self.config.physical_units:
                solar_luminosity = 3.826 * 10**33  # erg s^-1
                restframe_sed_erg_s_Hz = restframe_sed_Lsun_Hz[selected_wave_range] * solar_luminosity
                restframe_seds[i] = restframe_sed_erg_s_Hz.astype('float64')
            else:
                restframe_sed_Lsun_Hz = restframe_sed_Lsun_Hz[selected_wave_range]
                restframe_seds[i] = restframe_sed_Lsun_Hz.astype('float64')

        if self.comm is not None:  # pragma: no cover
            restframe_wavelengths = self.comm.gather(restframe_wavelengths)
            restframe_seds = self.comm.gather(restframe_seds)

            if self.rank != 0:  # pragma: no cover
                return None, None

            restframe_wavelengths = {k: v for a in restframe_wavelengths for k, v in a.items()}
            restframe_seds = {k: v for a in restframe_seds for k, v in a.items()}

        restframe_wavelengths = np.array([restframe_wavelengths[i] for i in range(len(tage))])
        restframe_seds = np.array([restframe_seds[i] for i in range(len(tage))])

        return restframe_wavelengths, restframe_seds

    def fit_model(self, compute_vega_mags=False, vactoair_flag=False, compute_light_ages=False, cloudy_dust=False,
                  agb_dust=1.0, tpagb_norm_type=2, dell=0.0, delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0,
                  sbss=0.0, pagb=1.0, pmetals=2.0, imf_upper_limit=120, imf_lower_limit=0.08, imf1=1.3, imf2=2.3,
                  imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, masscut=150.0, igm_factor=1.0, tau=1.0, const=0.0,
                  sf_start=0.0, sf_trunc=0.0, fburst=0.0, tburst=11.0, sf_slope=0.0,  dust_tesc=7.0, dust1=0.0,
                  dust2=0.0, dust_clumps=-99., frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, dust1_index=-1.0,
                  mwr=3.1, uvb=1.0, wgp1=1, wgp2=1, wgp3=1, duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5,
                  tabulated_sfh_files=None, tabulated_lsf_files=''):
        """
        Produce a creation model from which a rest-frame SED and photometry can be generated

        Parameters
        ----------
        See _get_rest_frame_seds function description for parameters meaning

        Returns
        -------
        model: ModelHandle
            ModelHandle storing the rest-frame SED model
        """

        self.run(compute_vega_mags=compute_vega_mags, vactoair_flag=vactoair_flag,
                 compute_light_ages=compute_light_ages, cloudy_dust=cloudy_dust,
                 agb_dust=agb_dust, tpagb_norm_type=tpagb_norm_type, dell=dell, delt=delt, redgb=redgb,
                 agb=agb, fcstar=fcstar, fbhb=fbhb, sbss=sbss, pagb=pagb, pmetals=pmetals,
                 imf_upper_limit=imf_upper_limit, imf_lower_limit=imf_lower_limit, imf1=imf1, imf2=imf2,
                 imf3=imf3, vdmc=vdmc, mdave=mdave, evtype=evtype, masscut=masscut, igm_factor=igm_factor, tau=tau,
                 const=const, sf_start=sf_start, sf_trunc=sf_trunc, fburst=fburst,
                 tburst=tburst, sf_slope=sf_slope, dust_tesc=dust_tesc, dust1=dust1, dust2=dust2, dust_clumps=dust_clumps,
                 frac_nodust=frac_nodust, frac_obrun=frac_obrun, dust_index=dust_index, dust1_index=dust1_index,
                 mwr=mwr, uvb=uvb, wgp1=wgp1, wgp2=wgp2, wgp3=wgp3, duste_gamma=duste_gamma, duste_umin=duste_umin,
                 duste_qpah=duste_qpah, tabulated_sfh_files=tabulated_sfh_files, tabulated_lsf_file=tabulated_lsf_files)
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self, compute_vega_mags=False, vactoair_flag=False, compute_light_ages=False, cloudy_dust=False,
            agb_dust=1.0, tpagb_norm_type=2, dell=0.0, delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0,
            sbss=0.0, pagb=1.0, pmetals=2.0, imf_upper_limit=120, imf_lower_limit=0.08, imf1=1.3, imf2=2.3,
            imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, masscut=150.0, igm_factor=1.0, tau=1.0, const=0.0, sf_start=0.0,
            sf_trunc=0.0, fburst=0.0, tburst=11.0, sf_slope=0.0, dust_tesc=7.0, dust1=0.0,
            dust2=0.0, dust_clumps=-99., frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, dust1_index=-1.0,
            mwr=3.1, uvb=1.0, wgp1=1, wgp2=1, wgp3=1, duste_gamma=0.01, duste_umin=1.0, duste_qpah=3.5,
            tabulated_sfh_files=None, tabulated_lsf_files=''):
        """
        Run method. It Calls `StellarPopulation` from FSPS to create a galaxy rest-frame SED.

        Parameters
        ----------
        See _get_rest_frame_seds function description for parameters meaning

        Notes
        -----
        Puts the rest-frame SED into the data store under this stages 'model' tag.

        The units of the resulting rest-frame SED is solar luminosity per Hertz. If the user provides a tabulated
        star-formation history in units of Msun/yr then, the luminosity refers to that emitted by the formed mass
        at the time of observation. Using the implemented parametric star-formation histories leads to a luminosity
        normalised to unit solar mass.

        Returns
        -------

        """

        data = self.get_data('input')

        redshifts = data[self.config.redshifts_key][()]
        ages = data[self.config.stellar_ages_key][()]
        metallicities = data[self.config.stellar_metallicities_key][()]
        velocity_dispersions = data[self.config.velocity_dispersions_key][()]
        gas_ionizations = data[self.config.gas_ionizations_key][()]
        gas_metallicities = data[self.config.gas_metallicities_key][()]

        if self.config.sfh_type == 3:
            tabulated_sfh_files = data[self.config.tabulated_sfh_key][()]
        elif (self.config.sfh_type == 1) | (self.config.sfh_type == 4):
            tau = data[self.config.tau_model_key][()][:, 0]
            const = data[self.config.tau_model_key][()][:, 1]
            sf_start = data[self.config.tau_model_key][()][:, 2]
            sf_trunc = data[self.config.tau_model_key][()][:, 3]
            fburst = data[self.config.tau_model_key][()][:, 4]
            tburst = data[self.config.tau_model_key][()][:, 5]
        elif self.config.sfh_type == 5:
            tau = data[self.config.tau_model_key][()][:, 0]
            const = data[self.config.tau_model_key][()][:, 1]
            sf_start = data[self.config.tau_model_key][()][:, 2]
            sf_trunc = data[self.config.tau_model_key][()][:, 3]
            fburst = data[self.config.tau_model_key][()][:, 4]
            tburst = data[self.config.tau_model_key][()][:, 5]
            sf_slope = data[self.config.tau_model_key][()][:, 6]
        else:
            raise ValueError

        if self.config.dust_type == 2:
            dust2 = data[self.config.dust_diffuse_key][()]
            dust1 = np.zeros_like(dust2)
        elif self.config.dust_type == 4:
            dust1 = data[self.config.dust_birth_cloud_key][()]
            dust2 = data[self.config.dust_diffuse_key][()]
            dust1_index = data[self.config.dust_powerlaw_modifier_key][()]
        else:
            print('Using default values for dust_type={}'.format(self.config.dust_type))

        if self.config.add_dust_emission is True:
            duste_gamma = data[self.config.dust_emission_gamma_key][()]
            duste_umin = data[self.config.dust_emission_umin_key][()]
            duste_qpah = data[self.config.dust_emission_qpah_key][()]

        if self.config.smooth_lsf:
            tabulated_lsf_files = data[self.config.tabulated_lsf_key][()]

        frac_bol_lum_agn = data[self.config.fraction_agn_bol_lum_key][()]
        agn_torus_opt_depths = data[self.config.agn_torus_opt_depth_key][()]

        wavelengths, restframe_seds = self._get_rest_frame_seds(compute_vega_mags=compute_vega_mags,
                                                                vactoair_flag=vactoair_flag,
                                                                zcontinuous=self.config.zcontinuous,
                                                                add_agb_dust_model=self.config.add_agb_dust_model,
                                                                add_dust_emission=self.config.add_dust_emission,
                                                                add_igm_absorption=self.config.add_igm_absorption,
                                                                add_neb_emission=self.config.add_neb_emission,
                                                                add_neb_continuum=self.config.add_neb_continuum,
                                                                add_stellar_remnants=self.config.add_stellar_remnants,
                                                                compute_light_ages=compute_light_ages,
                                                                nebemlineinspec=self.config.nebemlineinspec,
                                                                smooth_velocity=self.config.smooth_velocity,
                                                                smooth_lsf=self.config.smooth_lsf,
                                                                cloudy_dust=cloudy_dust, agb_dust=agb_dust,
                                                                tpagb_norm_type=tpagb_norm_type, dell=dell, delt=delt,
                                                                redgb=redgb, agb=agb, fcstar=fcstar, fbhb=fbhb,
                                                                sbss=sbss, pagb=pagb, zred=redshifts,
                                                                zmet=metallicities, logzsol=metallicities,
                                                                pmetals=pmetals, imf_type=self.config.imf_type,
                                                                imf_upper_limit=imf_upper_limit,
                                                                imf_lower_limit=imf_lower_limit, imf1=imf1, imf2=imf2,
                                                                imf3=imf3, vdmc=vdmc, mdave=mdave, evtype=evtype,
                                                                masscut=masscut, sigma_smooth=velocity_dispersions,
                                                                min_wave_smooth=self.config.min_wavelength,
                                                                max_wave_smooth=self.config.max_wavelength,
                                                                gas_logu=gas_ionizations, gas_logz=gas_metallicities,
                                                                igm_factor=igm_factor, sfh=self.config.sfh_type,
                                                                tau=tau, const=const, sf_start=sf_start,
                                                                sf_trunc=sf_trunc, tage=ages,
                                                                fburst=fburst, tburst=tburst, sf_slope=sf_slope,
                                                                dust_type=self.config.dust_type, dust_tesc=dust_tesc,
                                                                dust1=dust1, dust2=dust2, dust_clumps=dust_clumps,
                                                                frac_nodust=frac_nodust, frac_obrun=frac_obrun,
                                                                dust_index=dust_index, dust1_index=dust1_index,
                                                                mwr=mwr, uvb=uvb, wgp1=wgp1, wgp2=wgp2, wgp3=wgp3,
                                                                duste_gamma=duste_gamma, duste_umin=duste_umin,
                                                                duste_qpah=duste_qpah,
                                                                fagn=frac_bol_lum_agn, agn_tau=agn_torus_opt_depths,
                                                                tabulated_sfh_files=tabulated_sfh_files,
                                                                tabulated_lsf_files=tabulated_lsf_files)

        if self.rank == 0:
            rest_frame_sed_models = {'wavelength': wavelengths[0], 'restframe_seds': restframe_seds,
                                     'redshifts': redshifts} # (n_galaxies, n_wavelengths) = (100000000, 4096)
            self.add_data('model', rest_frame_sed_models)
