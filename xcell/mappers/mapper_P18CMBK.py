from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt
from .utils import rotate_mask


class MapperP18CMBK(MapperBase):
    """
    Mapper for the Planck 18 convergence map. 
    """
    def __init__(self, config):
        """
        config - dict
        {'file_klm':'COM_Lensing_4096_R3.00/MV/dat_klm.fits',
         'file_mask':'COM_Lensing_4096_R3.00/mask.fits.gz',
         'file_noise':'COM_Lensing_4096_R3.00/MV/nlkk.dat',
         'mask_aposize': 0.2,
         'mask_apotype': 'C1',
         'mask_name': 'mask_CMBK',
         'nside':4096}
        """
        self._get_defaults(config)
        self.mask_apotype = config.get('mask_apotype', 'C1')
        self.mask_aposize = config.get('mask_aposize', 0.2)

        self.noise = None

        # Galactic-to-celestial coordinate rotator
        self.rot = self._get_rotator('G')

        # Defaults
        self.signal_map = None
        self.nl_coupled = None
        self.cl_fid = None

    def get_signal_map(self):
        """
        Returns mapper's signal map. \
        If the highest multipole is larger than \
        the one allowed by the resolution \
        Args:
            None
        Returns:
            signal_map (Array)
        """
        if self.signal_map is None:
            # Read alms
            self.klm, lmax = hp.read_alm(self.config['file_klm'],
                                         return_mmax=True)
            if self.rot is not None:
                self.klm = self.rot.rotate_alm(self.klm)
            # Filter if lmax is too large
            if lmax > 3*self.nside-1:
                fl = np.ones(lmax+1)
                fl[3*self.nside:] = 0
                self.klm = hp.almxfl(self.klm, fl, inplace=True)
            self.signal_map = np.array([hp.alm2map(self.klm, self.nside)])
        return self.signal_map

    def _get_mask(self):
        """
        Returns the mapper's mask \
        after applying the neccesary coordinate \
        rotations and apodizing. \
        Args:
            None
        Returns:
            msk (Array)
        """
        msk = hp.read_map(self.config['file_mask'],
                          dtype=float)
        msk = rotate_mask(msk, self.rot, binarize=True)
        # Apodize
        msk = nmt.mask_apodization(msk, self.mask_aposize,
                                   self.mask_apotype)
        msk = hp.ud_grade(msk, nside_out=self.nside)
        return msk

    def get_nl_coupled(self):
        """
        Returns the coupled noise power \
        spectrum of the mapper. \
        It loads the decoupled noise power spectrum \
        and then it multiplies it by the mean of the \
        squared mask to couple it. \
        Args:
            None
        Returns:
            nl_coupled (Array)
        """
        if self.nl_coupled is None:
            ell = self.get_ell()
            noise = self._get_noise()
            nl = noise[1]
            nl = interp1d(noise[0], nl, bounds_error=False,
                          fill_value=(nl[0], nl[-1]))(ell)

            # The Nl in the file is decoupled. To "couple" it, multiply by the
            # mean of the squared mask. This will account for the factor that
            # will be divided for in the covariance.
            nl *= np.mean(self.get_mask()**2.)
            self.nl_coupled = np.array([nl])
        return self.nl_coupled

    def get_cl_fiducial(self):
        """
        Returns the signal power spectrum \
        of the mapper. \
        Args:
            None
        Returns:
            cl_fid (Array)
        """
        if self.cl_fid is None:
            ell = self.get_ell()
            noise = self._get_noise()
            cl = noise[2] - noise[1]
            cl = interp1d(noise[0], cl, bounds_error=False,
                          fill_value=(cl[0], cl[-1]))(ell)
            self.cl_fid = np.array([cl])
        return self.cl_fid

    def _get_noise(self):
        """
        Returns the decoupled noise power spectrum of the \
        auto-correlation of the covergence map. \
        Args:
            None
        Returns:
            [l (Array): multipole list, 
             Nl (Array): noise power spectrum, 
             Nl+Cl (Array): noise + signal power spectrum] (Array)
        """
        if self.noise is None:
            # Read noise file. Column order is: ['l', 'Nl', 'Nl+Cl']
            self.noise = np.loadtxt(self.config['file_noise'], unpack=True)

        return self.noise

    def get_dtype(self):
         """
        Returns the type of the mapper. \
        
        Args:
            None
        Returns:
            mapper_type (String)
        """
        return 'cmb_convergence'

    def get_spin(self):
        """
        Returns the spin of the mapper. \
        
        Args:
            None
        Returns:
            spin (Int)
        """
        return 0
