import healpy as hp
import numpy as np
from .utils import rotate_map, subtract_mono_and_dipole
from .mapper_P15tSZ import MapperP15tSZ

class BaseMapperNPIPEtSZ(MapperP15tSZ):
    """
    Base mapper for NPIPE PR4 tSZ maps
    """

    map_name = None

    def __init__(self, config):
        super().__init__(config)
        self.remove_dipole = config.get("remove_dipole", True)

    def __iter__(self):
        """
        Allows the mapper to be
        iterated over its attributes
        """
        for attr, value in self.__dict__.items():
            yield attr, value

    def _get_signal_map(self):
        """
        Read the map file and
        optionally remove the dipole from it
        """
        signal_map = hp.read_map(self.file_map)
        signal_map[signal_map == hp.UNSEEN] = 0.0
        signal_map[np.isnan(signal_map)] = 0.0
        ps_mask = self._get_ps_mask()
        signal_map *= ps_mask
        signal_map = rotate_map(signal_map, self.rot)
        if self.remove_dipole:
            field = self.gp_mask_modes[self.gp_mask_mode]
            gp_mask = hp.read_map(self.file_gp_mask, field)
            mask = gp_mask * ps_mask
            signal_map = subtract_mono_and_dipole(signal_map, mask)
        signal_map = np.array([hp.ud_grade(signal_map, nside_out=self.nside)])
        return signal_map

    def _generate_hm_maps(self):
        """
        Read HM1/HM2 maps from either one
        or different fits files.
        """
        maps = []
        ps_mask = self._get_ps_mask()
        for id, split in enumerate(["hm1", "hm2"]):
            file_name = getattr(self, f"file_{split}")
            # If only one file is given, we read the map
            # from the same fits file
            if file_name == self.file_map:
                m = hp.read_map(file_name, id+1)
            else:
                m = hp.read_map(file_name)
            m *= ps_mask
            m = rotate_map(m, self.rot)
            if self.remove_dipole:
                field = self.gp_mask_modes[self.gp_mask_mode]
                gp_mask = hp.read_map(self.file_gp_mask, field)
                mask = gp_mask * ps_mask
                m = subtract_mono_and_dipole(m, mask)
            m = hp.ud_grade(m, nside_out=self.nside)
            maps.append(m)
        return np.array(maps)

class MapperNPIPEtSZ_mccarthy(BaseMapperNPIPEtSZ):
    """
    Mapper for the Planck PR4 tSZ map
    discussed in McCarthy et al.
    arXiv:2307.01043
    """
    map_name = "Planck__NPIPEtSZ_mccarthy"

class MapperNPIPEtSZ_chandran(BaseMapperNPIPEtSZ):
    """
    Mapper for the Planck PR4 tSZ map
    discussed in Chandran et al.
    arXiv:2305.10193
    """
    map_name = "Planck__NPIPEtSZ_chandran"