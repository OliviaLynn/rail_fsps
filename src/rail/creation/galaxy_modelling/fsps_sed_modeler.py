from rail.creation.engine import Modeler
from rail.core.stage import RailStage
from rail.core.data import ModelHandle
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
                          add_neb_emission=Param(bool, False, msg='Turn on/off nebular emission model based on Cloudy'),
                          add_neb_continuum=Param(bool, False, msg='Turn on/off nebular continuum component'),
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
                          galaxy_metallicities=Param(str, 'galaxy_metallicities.npy',
                                                     msg='npy file containing the metallicity values in units of '
                                                     'log10(Z / Zsun)'),
                          galaxy_ages=Param(str, 'galaxy_ages.npy', msg='npy file containing the age values in units'
                                                                        'of Gyr'),
                          galaxy_vel_disp=Param(str, 'None', msg='Default to None, otherwise npy file containing the '
                                                                 'velocity dispersion values in units of km/s'),
                          gas_ionization_params=Param(str, 'None', msg='Default to None, npy file containing the gas '
                                                                       'ionization parameters values'),
                          gas_metallicity_params=Param(str, 'None', msg='Default to None, npy file containing the gas '
                                                                        'metallicities values in units of'
                                                                        ' log10(Z / Zsun)'),
                          dust_birth_cloud_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                         'dust parameters describing the attenuation '
                                                                         'of young stellar light'),
                          dust_diffuse_params=Param(str, 'None', msg='Default to None, npy file containing the '
                                                                     'dust parameters describing the attenuation of '
                                                                     'old stellar light'),
                          dust_powerlaw_modifier_params=Param(str, 'None', msg='Default to None, npy file containing'
                                                                               'the power-law modifiers to the shape '
                                                                               'of the Calzetti et al. (2000) '
                                                                               'attenuation curve'),
                          dust_emission_gamma_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                            ' Relative contributions of dust heated at'
                                                                            ' Umin, parameter of Draine and Li (2007) '
                                                                            'dust emission model'),
                          dust_emission_umin_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                           ' Minimum radiation field strengths, '
                                                                           'parameter of Draine and Li (2007) dust '
                                                                           'emission model'),
                          dust_emission_qpah_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                           ' Grain size distributions in mass in PAHs, '
                                                                           'parameter of Draine and Li (2007) dust '
                                                                           'emission model'),
                          fraction_agn_bol_lum_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                             ' Fractional contributions of AGN wrt '
                                                                             'stellar bolometric luminosity'),
                          agn_torus_opt_depth_params=Param(str, 'None', msg='Default to None, npy file containing the'
                                                                            'Optical depths of the AGN dust torii'),
                          physical_units=Param(bool, False), msg='False (True) for rest-frame spectra in units of'
                                                                 'Lsun/Hz (erg/s/Hz)')

    # inputs = [("input", DataHandle)]
    # outputs = [("model", ModelHandle)]
    outputs = [("model", ModelHandle)]

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

        self.ages = np.load(self.config.galaxy_ages)
        self.metallicities = np.load(self.config.galaxy_metallicities)

        if self.config.galaxy_vel_disp == 'None':
            self.velocity_dispersions = np.full(len(self.ages), 0.)  # default value in km/s in FSPS
        else:
            self.velocity_dispersions = np.load(self.config.galaxy_vel_disp)

        if self.config.gas_ionization_params == 'None':
            self.gas_ionizations = np.full(len(self.ages), -2.)  # default value in FSPS
        else:
            self.gas_ionizations = np.load(self.config.gas_ionization_params)

        if self.config.gas_metallicity_params == 'None':
            self.gas_metallicities = self.metallicities.copy()
        else:
            self.gas_metallicities = np.load(self.config.gas_metallicity_params)

        if self.config.dust_birth_cloud_params == 'None':
            self.dust_birth_cloud = np.full(len(self.ages), 0.)  # default value in FSPS
        else:
            self.dust_birth_cloud = np.load(self.config.dust_birth_cloud_params)

        if self.config.dust_diffuse_params == 'None':
            self.dust_diffuse = np.full(len(self.ages), 0.)  # default value in FSPS
        else:
            self.dust_diffuse = np.load(self.config.dust_diffuse_params)

        if self.config.dust_powerlaw_modifier_params == 'None':
            self.dust_powerlaw_modifier = np.full(len(self.ages), -1.)  # default value in FSPS
        else:
            self.dust_powerlaw_modifier = np.load(self.config.dust_powerlaw_modifier_params)

        if self.config.dust_emission_gamma_params == 'None':
            self.dust_emission_gamma = np.full(len(self.ages), 0.01)  # default value in FSPS
        else:
            self.dust_emission_gamma = np.load(self.config.dust_emission_gamma_params)

        if self.config.dust_emission_umin_params == 'None':
            self.dust_emission_umin = np.full(len(self.ages), 1.0)  # default value in FSPS
        else:
            self.dust_emission_umin = np.load(self.config.dust_emission_umin_params)

        if self.config.dust_emission_qpah_params == 'None':
            self.dust_emission_qpah = np.full(len(self.ages), 3.5)  # default value in FSPS
        else:
            self.dust_emission_qpah = np.load(self.config.dust_emission_qpah_params)

        if self.config.fraction_agn_bol_lum_params == 'None':
            self.fraction_agn_bol_lum = np.full(len(self.ages), 0.)  # default value in FSPS
        else:
            self.fraction_agn_bol_lum = np.load(self.config.fraction_agn_bol_lum_params)

        if self.config.agn_torus_opt_depth_params == 'None':
            self.agn_torus_opt_depth = np.full(len(self.ages), 10)  # default value in FSPS
        else:
            self.agn_torus_opt_depth = np.load(self.config.agn_torus_opt_depth_params)

    def _get_rest_frame_seds(self, compute_vega_mags=False, vactoair_flag=False, add_agb_dust_model=True,
                             add_dust_emission=True, add_igm_absorption=False, add_stellar_remnants=True,
                             compute_light_ages=False, cloudy_dust=False, agb_dust=1.0, tpagb_norm_type=2,
                             dell=0.0, delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0, sbss=0.0, pagb=1.0,
                             zmet=1, pmetals=2.0, imf_upper_limit=120, imf_lower_limit=0.08,
                             imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, use_wr_spectra=1,
                             logt_wmb_hot=0.0, masscut=150.0, igm_factor=1.0, tau=np.ones(1),
                             const=np.zeros(1), sf_start=np.zeros(1), sf_trunc=np.zeros(1), fburst=np.zeros(1),
                             tburst=np.ones(1), sf_slope=np.zeros(1), dust_tesc=7.0, dust_clumps=-99.,
                             frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, mwr=3.1, uvb=1.0, wgp1=1,
                             wgp2=1, wgp3=1, physical_units=True, zred=None, tabulated_sfh_file=None,
                             tabulated_lsf_file=None):
        """

        Parameters
        ----------
        compute_vega_mags: bool
            Default to False for AB magnitudes, True for Vega magnitudes
        vactoair_flag: bool
            Defaul to False for vacuum wavelengths, True for air wavelengths
        add_agb_dust_model: bool
            Default to True, adding AGB circumstellar dust contribution to SED
        add_dust_emission: bool
            Default to True, adding dust emission contribution to SED
        add_igm_absorption: bool
            Default to False, adding IGM absorption contribution to SED
        add_stellar_remnants: bool
            Default to True, adding stellar remnants contribution to stellar mass
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

        if len(tau) == 1:
            tau = np.full(len(self.ages), tau[0])
            const = np.full(len(self.ages), const[0])
            sf_start = np.full(len(self.ages), sf_start[0])
            sf_trunc = np.full(len(self.ages), sf_trunc[0])
            fburst = np.full(len(self.ages), fburst[0])
            tburst = np.full(len(self.ages), tburst[0])
            sf_slope = np.full(len(self.ages), sf_slope[0])

        if zred is None:
            redshifts = np.full(len(self.ages), 0)
        else:
            redshifts = np.load(zred)

        restframe_wavelengths = {}
        restframe_seds = {}

        for i in self.split_tasks_by_rank(range(len(self.ages))):
            sp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags, vactoair_flag=vactoair_flag,
                                        zcontinuous=self.config.zcontinuous, add_agb_dust_model=add_agb_dust_model,
                                        add_dust_emission=add_dust_emission, add_igm_absorption=add_igm_absorption,
                                        add_neb_emission=self.config.add_neb_emission,
                                        add_neb_continuum=self.config.add_neb_continuum,
                                        add_stellar_remnants=add_stellar_remnants,
                                        compute_light_ages=compute_light_ages,
                                        nebemlineinspec=self.config.nebemlineinspec,
                                        smooth_velocity=self.config.smooth_velocity, smooth_lsf=self.config.smooth_lsf,
                                        cloudy_dust=cloudy_dust, agb_dust=agb_dust, tpagb_norm_type=tpagb_norm_type,
                                        dell=dell, delt=delt, redgb=redgb, agb=agb, fcstar=fcstar, fbhb=fbhb,
                                        sbss=sbss, pagb=pagb, zred=redshifts[i], zmet=zmet,
                                        logzsol=self.metallicities[i],
                                        pmetals=pmetals, imf_type=self.config.imf_type,
                                        imf_upper_limit=imf_upper_limit, imf_lower_limit=imf_lower_limit,
                                        imf1=imf1, imf2=imf2, imf3=imf3, vdmc=vdmc, mdave=mdave, evtype=evtype,
                                        use_wr_spectra=use_wr_spectra, logt_wmb_hot=logt_wmb_hot, masscut=masscut,
                                        sigma_smooth=self.velocity_dispersions[i],
                                        min_wave_smooth=self.config.min_wavelength,
                                        max_wave_smooth=self.config.max_wavelength,
                                        gas_logu=self.gas_ionizations[i], gas_logz=self.gas_metallicities[i],
                                        igm_factor=igm_factor, sfh=self.config.sfh_type, tau=tau[i], const=const[i],
                                        sf_start=sf_start[i], sf_trunc=sf_trunc[i], tage=self.ages[i], fburst=fburst[i],
                                        tburst=tburst[i], sf_slope=sf_slope[i], dust_type=self.config.dust_type,
                                        dust_tesc=dust_tesc, dust1=self.dust_birth_cloud[i], dust2=self.dust_diffuse[i],
                                        dust_clumps=dust_clumps, frac_nodust=frac_nodust, frac_obrun=frac_obrun,
                                        dust_index=dust_index, dust1_index=self.dust_powerlaw_modifier[i],
                                        mwr=mwr, uvb=uvb, wgp1=wgp1, wgp2=wgp2, wgp3=wgp3,
                                        duste_gamma=self.dust_emission_gamma[i], duste_umin=self.dust_emission_umin[i],
                                        duste_qpah=self.dust_emission_qpah[i], fagn=self.fraction_agn_bol_lum[i],
                                        agn_tau=self.agn_torus_opt_depth[i])

            if self.config.sfh_type == 3:

                if self.config.zcontinuous == 3:
                    age_array, sfr_array, metal_array = np.loadtxt(tabulated_sfh_file[i], usecols=(0, 1, 2),
                                                                   unpack=True)
                    sp.set_tabular_sfh(age_array, sfr_array, Z=metal_array)
                elif self.config.zcontinuous == 1:
                    age_array, sfr_array = np.loadtxt(tabulated_sfh_file[i], usecols=(0, 1), unpack=True)
                    sp.set_tabular_sfh(age_array, sfr_array, Z=None)
                else:
                    raise ValueError

            if self.config.smooth_lsf:
                assert self.config.smooth_velocity is True, 'lsf smoothing only works if smooth_velocity is True'
                lsf_values = np.loadtxt(tabulated_lsf_file, usecols=(0, 1))
                wave = lsf_values[:, 0]  # pragma: no cover
                sigma = lsf_values[:, 1]  # pragma: no cover
                sp.set_lsf(wave, sigma, wmin=self.config.min_wavelength,
                           wmax=self.config.max_wavelength)  # pragma: no cover

            restframe_wavelength, restframe_sed_Lsun_Hz = sp.get_spectrum(tage=self.ages[i], peraa=False)

            selected_wave_range = np.where((restframe_wavelength >= self.config.min_wavelength) &
                                           (restframe_wavelength <= self.config.max_wavelength))
            restframe_wavelength = restframe_wavelength[selected_wave_range]
            restframe_wavelengths[i] = restframe_wavelength

            if physical_units:
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

        restframe_wavelengths = np.array([restframe_wavelengths[i] for i in range(len(self.ages))])
        restframe_seds = np.array([restframe_seds[i] for i in range(len(self.ages))])

        return restframe_wavelengths, restframe_seds

    def fit_model(self, compute_vega_mags=False, vactoair_flag=False, add_agb_dust_model=True,
                  add_dust_emission=True, add_igm_absorption=False, add_stellar_remnants=True,
                  compute_light_ages=False, cloudy_dust=False, agb_dust=1.0, tpagb_norm_type=2,
                  dell=0.0, delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0, sbss=0.0, pagb=1.0,
                  zmet=1, pmetals=2.0, imf_upper_limit=120, imf_lower_limit=0.08,
                  imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, use_wr_spectra=1,
                  logt_wmb_hot=0.0, masscut=150.0, igm_factor=1.0, tau=np.ones(1),
                  const=np.zeros(1), sf_start=np.zeros(1), sf_trunc=np.zeros(1), fburst=np.zeros(1),
                  tburst=np.ones(1), sf_slope=np.zeros(1), dust_tesc=7.0, dust_clumps=-99.,
                  frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, mwr=3.1, uvb=1.0, wgp1=1,
                  wgp2=1, wgp3=1, zred=None,
                  tabulated_sfh_files=None, tabulated_lsf_file=''):
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
                 add_agb_dust_model=add_agb_dust_model, add_dust_emission=add_dust_emission,
                 add_igm_absorption=add_igm_absorption, add_stellar_remnants=add_stellar_remnants,
                 compute_light_ages=compute_light_ages, cloudy_dust=cloudy_dust, agb_dust=agb_dust,
                 tpagb_norm_type=tpagb_norm_type, dell=dell, delt=delt, redgb=redgb, agb=agb,
                 fcstar=fcstar, fbhb=fbhb, sbss=sbss, pagb=pagb,
                 zmet=zmet, pmetals=pmetals, imf_upper_limit=imf_upper_limit,
                 imf_lower_limit=imf_lower_limit, imf1=imf1, imf2=imf2, imf3=imf3, vdmc=vdmc, mdave=mdave,
                 evtype=evtype, use_wr_spectra=use_wr_spectra, logt_wmb_hot=logt_wmb_hot, masscut=masscut,
                 igm_factor=igm_factor, tau=tau, const=const, sf_start=sf_start, sf_trunc=sf_trunc, fburst=fburst,
                 tburst=tburst, sf_slope=sf_slope, dust_tesc=dust_tesc, dust_clumps=dust_clumps,
                 frac_nodust=frac_nodust, frac_obrun=frac_obrun, dust_index=dust_index, mwr=mwr, uvb=uvb, wgp1=wgp1,
                 wgp2=wgp2, wgp3=wgp3, zred=zred, tabulated_sfh_files=tabulated_sfh_files,
                 tabulated_lsf_file=tabulated_lsf_file)
        self.finalize()
        model = self.get_handle("model")
        return model

    def run(self, compute_vega_mags=False, vactoair_flag=False, add_agb_dust_model=True,
            add_dust_emission=True, add_igm_absorption=False, add_stellar_remnants=True,
            compute_light_ages=False, cloudy_dust=False, agb_dust=1.0, tpagb_norm_type=2,
            dell=0.0, delt=0.0, redgb=1.0, agb=1.0, fcstar=1.0, fbhb=0.0, sbss=0.0, pagb=1.0,
            zmet=1, pmetals=2.0, imf_upper_limit=120, imf_lower_limit=0.08,
            imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, evtype=-1, use_wr_spectra=1,
            logt_wmb_hot=0.0, masscut=150.0, igm_factor=1.0, tau=np.ones(1),
            const=np.zeros(1), sf_start=np.zeros(1), sf_trunc=np.zeros(1), fburst=np.zeros(1),
            tburst=np.ones(1), sf_slope=np.zeros(1), dust_tesc=7.0, dust_clumps=-99.,
            frac_nodust=0.0, frac_obrun=0.0, dust_index=-0.7, mwr=3.1, uvb=1.0, wgp1=1,
            wgp2=1, wgp3=1, zred=None, tabulated_sfh_files=None, tabulated_lsf_file=''):
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

        if tabulated_sfh_files is None:
            tabulated_sfh_files = []

        wavelengths, restframe_seds = self._get_rest_frame_seds(compute_vega_mags=compute_vega_mags,
                                                                vactoair_flag=vactoair_flag,
                                                                add_agb_dust_model=add_agb_dust_model,
                                                                add_dust_emission=add_dust_emission,
                                                                add_igm_absorption=add_igm_absorption,
                                                                add_stellar_remnants=add_stellar_remnants,
                                                                compute_light_ages=compute_light_ages,
                                                                cloudy_dust=cloudy_dust, agb_dust=agb_dust,
                                                                tpagb_norm_type=tpagb_norm_type,
                                                                dell=dell, delt=delt, redgb=redgb, agb=agb,
                                                                fcstar=fcstar, fbhb=fbhb, sbss=sbss, pagb=pagb,
                                                                zmet=zmet, pmetals=pmetals,
                                                                imf_upper_limit=imf_upper_limit,
                                                                imf_lower_limit=imf_lower_limit,
                                                                imf1=imf1, imf2=imf2, imf3=imf3, vdmc=vdmc,
                                                                mdave=mdave, evtype=evtype,
                                                                use_wr_spectra=use_wr_spectra,
                                                                logt_wmb_hot=logt_wmb_hot, masscut=masscut,
                                                                igm_factor=igm_factor, tau=tau, const=const,
                                                                sf_start=sf_start, sf_trunc=sf_trunc, fburst=fburst,
                                                                tburst=tburst, sf_slope=sf_slope, dust_tesc=dust_tesc,
                                                                dust_clumps=dust_clumps, frac_nodust=frac_nodust,
                                                                frac_obrun=frac_obrun, dust_index=dust_index, mwr=mwr,
                                                                uvb=uvb, wgp1=wgp1, wgp2=wgp2, wgp3=wgp3,
                                                                physical_units=self.config.physical_units,
                                                                zred=zred,
                                                                tabulated_sfh_file=tabulated_sfh_files,
                                                                tabulated_lsf_file=tabulated_lsf_file)

        if self.rank == 0:
            rest_frame_sed_models = {'wavelength': wavelengths[0], 'restframe_seds': restframe_seds} # (n_galaxies, n_wavelengths) = (100000000, 4096)
            self.add_data('model', rest_frame_sed_models)
