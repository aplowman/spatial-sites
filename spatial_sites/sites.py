"""`spatial_sites.sites.py`

Module defining a Sites class that represents a set of points in space.

"""

import numbers
import warnings
import re
import copy

import numpy as np

from spatial_sites.utils import check_indices


def vector_direction_setter(obj, vector_direction):

    if vector_direction not in ['row', 'r', 'column', 'col', 'c']:
        msg = ('`vector_direction` must be specified as a string, either '
               '"row" (or "r") or "column" (or "col" or "c").')
        raise ValueError(msg)

    if vector_direction in ['col', 'c']:
        vector_direction = 'column'
    elif vector_direction == 'r':
        vector_direction = 'row'

    old_vec_dir = getattr(obj, '_vector_direction', None)
    if old_vec_dir:
        if vector_direction == old_vec_dir:
            msg = '`vector_direction` is already set to "{}"'
            warnings.warn(msg.format(vector_direction))

    obj._vector_direction = vector_direction


class SitesLabel(object):
    """Class to represent the labelling of a set of points in space.

    Attributes
    ----------
    name : str
    unique_values : ndarray
    values_idx : ndarray of int
    values : ndarray

    """

    def __init__(self, name, values=None, unique_values=None, values_idx=None):

        args = [values, unique_values, values_idx]
        msg = ('Specify either `values` or both `unique_values` and '
               '`values_idx`')
        if all([i is not None for i in args]) or all([i is None for i in args]):
            raise ValueError(msg)

        if values is not None and not isinstance(values, np.ndarray):
            values = np.array(values)

        if unique_values is not None and not isinstance(unique_values, np.ndarray):
            unique_values = np.array(unique_values)

        if values_idx is not None and not isinstance(values_idx, np.ndarray):
            values_idx = np.array(values_idx)

        if values is not None:
            # Get unique `values` and place indices in `values_idx`:
            unique_values, values_idx = np.unique(values, return_inverse=True)

        else:
            # Check unique values are all unique:
            if len(np.unique(unique_values)) != len(unique_values):
                msg = ('Not all of the values in `unique_values` are unique.')
                raise ValueError(msg)

            # Check all `values_idx` do index `unique_values`:
            check_indices(unique_values, values_idx)

        self._validate_name(name)
        self.name = name
        self.unique_values = unique_values
        self.values_idx = values_idx

    @property
    def values(self):
        return self.unique_values[self.values_idx]

    @property
    def dtype(self):
        return self.unique_values.dtype

    def __len__(self):
        return len(self.values)

    def __copy__(self):
        out = SitesLabel(
            name=self.name,
            unique_values=np.copy(self.unique_values),
            values_idx=np.copy(self.values_idx)
        )
        return out

    def _validate_name(self, name):
        """Ensure name is safe to use as an object attribute."""
        pattern = r'^(?![0-9])[a-zA-Z0-9_]+$'
        if not re.match(pattern, name):
            msg = ('SitesLabel name "{}" is not valid since it cannot be '
                   'used as an object attribute name. Names must match the '
                   'regular expression "{}".')
            raise ValueError(msg.format(name, pattern))


class Sites(object):
    """Class to represent a set of points in space.

    Attributes
    ----------
    sites : ndarray
    dimension : int
    vector_direction : str
    labels : dict of (str : (dict or SitesLabel))

    """

    def __init__(self, sites, labels=None, vector_direction='column',
                 dimension=3):

        self.vector_direction = vector_direction
        self._sites = self._validate(sites, self.vector_direction, dimension)
        self._dimension = dimension
        self._labels = self._init_labels(labels)

    def __setattr__(self, name, value):
        """Overridden method to prevent reassigning label attributes."""

        if getattr(self, '_labels', None) and name in self._labels:
            msg = 'Can\'t set attribute "{}"'.format(name)
            raise AttributeError(msg)

        # Set all other attributes as normal:
        super().__setattr__(name, value)

    def __len__(self):
        """Get how many sites there are in this Sites objects."""
        return self._sites.shape[1]

    def __copy__(self):
        out = Sites(
            sites=np.copy(self.sites),
            vector_direction=self.vector_direction,
            labels=copy.deepcopy(self.labels),
            dimension=self.dimension,
        )
        return out

    def __add__(self, obj):

        out = self.copy()
        if isinstance(obj, Sites):
            out += obj
        else:
            out._sites += self._validate_trans_vector(obj)

        return out

    def __radd__(self, obj):

        if not isinstance(obj, type(self)):
            out = self.__add__(obj)
            return out

    def __iadd__(self, obj):

        if isinstance(obj, Sites):
            # Concatenate sites:
            self._validate_concat(self, obj)

            new_labs = {}
            for lab_name, sites_lab in self.labels.items():

                new_lab_vals = np.hstack([sites_lab.values,
                                          obj.labels[lab_name].values])

                sites_lab_new = SitesLabel(name=lab_name, values=new_lab_vals)
                new_labs.update({
                    lab_name: sites_lab_new,
                })
                super().__setattr__(lab_name, sites_lab_new.values)

            new_sites = np.hstack([self._sites, obj._sites])
            self._sites = new_sites
            self._labels = new_labs

        else:
            # Add a translation vector:
            self._sites += self._validate_trans_vector(obj)

        return self

    def __sub__(self, vector):

        out = self.copy()
        out._sites -= self._validate_trans_vector(vector)

        return out

    def __rsub__(self, vector):

        out = self.copy()
        out._sites = self._validate_trans_vector(vector) - out._sites

        return out

    def __isub__(self, vector):

        self._sites -= self._validate_trans_vector(vector)

        return self

    def _init_labels(self, labels):
        """Set labels as attributes for easy access."""

        label_objs = {}
        for k, v in (labels or {}).items():

            if isinstance(v, SitesLabel):
                sites_label = v

            else:
                msg = ('Specify site labels as either a single list/tuple of '
                       'values, or as a list/tuple of length two, whose first '
                       'element is a list/tuple of unique values, and whose '
                       'second element is a list/tuple of indices that index '
                       'the first element.')
                values = None
                unique_values = None
                values_idx = None

                if len(v) == 2:
                    if isinstance(v[0], (np.ndarray, list, tuple)):
                        unique_values, values_idx = v
                    else:
                        raise ValueError(msg)
                else:
                    values = v

                sites_label = SitesLabel(
                    k,
                    values=values,
                    unique_values=unique_values,
                    values_idx=values_idx
                )

            msg = ('Length of site labels named "{}" ({}) does not match '
                   'the number of sites ({}).')
            vals = sites_label.values
            if len(vals) != len(self):
                raise ValueError(msg.format(k, len(vals), len(self)))

            setattr(self, k, vals)
            label_objs.update({k: sites_label})

        return label_objs

    def _validate(self, sites, vector_direction, dimension):
        """Validate inputs."""

        if dimension not in [2, 3]:
            msg = '`dimension` must be an integer: 2 or 3.'
            raise ValueError(msg)

        if not isinstance(sites, np.ndarray):
            sites = np.array(sites)

        if sites.ndim != 2:
            raise ValueError('`sites` must be a 2D array.')

        vec_len_idx = 0 if vector_direction == 'column' else 1
        vec_len = sites.shape[vec_len_idx]

        if vec_len != dimension:
            msg = ('The length of {}s in `sites` ({}) must be equal to '
                   '`dimension` ({}). Change `vector_direction` to "{}" if '
                   'you would like an individual site to be represented as '
                   'a {}-vector')
            non_vec_dir = 'row' if vector_direction == 'column' else 'column'
            raise ValueError(
                msg.format(
                    vector_direction,
                    vec_len,
                    dimension,
                    non_vec_dir,
                    non_vec_dir,
                )
            )

        if self.vector_direction == 'row':
            return sites.T
        else:
            return sites

    def _validate_label_filter(self, **kwargs):
        """Validation for the `index` method."""

        if not self.labels:
            raise ValueError(
                'No labels are associated with this Sites object.')

        if not kwargs:
            msg = ('Provide a label condition to filter the sites. Available '
                   'labels are: {}')
            raise ValueError(msg.format(list(self.labels.keys())))

        if len(kwargs) > 1:
            msg = 'Only one label condition is currently supported by `whose`.'
            raise NotImplementedError(msg)

        for match_label, match_val in kwargs.items():
            try:
                getattr(self, match_label)
            except AttributeError as err:
                msg = 'No Sites label called "{}" was found.'
                raise ValueError(msg.format(match_label))

            return match_label, match_val

    def _validate_trans_vector(self, vector):
        """Validate that an input vector is suitable for translation.

        Parameters
        ---------
        vector : list or ndarray

        Returns
        -------
        ndarray of shape (self.dimension, 1)

        """

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        if len(vector.shape) > 1:
            vector = np.squeeze(vector)

        if vector.shape != (self.dimension, ):
            msg = ('Cannot translate coordinates with dimension {} by a '
                   'vector with shape {}.')
            raise ValueError(msg.format(self.dimension, vector.shape))

        return vector[:, None]

    @staticmethod
    def _validate_concat(*sites):
        """Validate two or more Sites objects are compatible for concatenation.

        args : Sites objects

        """

        if len(sites) < 2:
            msg = ('At least two `Sites` objects must be supplied.')
            raise ValueError(msg)

        labs = {
            k: v.dtype
            for k, v in sites[0].labels.items()
        }
        dim = sites[0].dimension
        vec_dir = sites[0].vector_direction

        for i in sites[1:]:

            # Check for same `dimension`s
            if i.dimension != dim:
                msg = ('Incompatible `Sites` objects: inconsistent '
                       '`dimension`s.')
                raise ValueError(msg)

            # Check for same `vector_direction`s:
            if i.vector_direction != vec_dir:
                msg = ('Incompatible `Sites` objects: inconsistent '
                       '`vector_direction`s.')
                raise ValueError(msg)

            labs_i = i.labels

            # Check for same label `name`s:
            if not (set(labs.keys()) | set(labs_i.keys())) == set(labs_i.keys()):
                msg = 'Incompatible `Sites` objects: different labels exist.'
                raise ValueError(msg)

            # Check for same `dtype`s:
            for k, v in labs_i.items():
                if not (np.can_cast(labs[k], v.dtype) or
                        np.can_cast(v.dtype, labs[k])):
                    msg = ('Incompatible `Sites` objects: labels named "{}" '
                           'have uncastable `dtype`s: {} and {}')
                    raise ValueError(msg.format(k, labs[k], v.dtype))

    @property
    def labels(self):
        return self._labels

    @property
    def dimension(self):
        return self._dimension

    @property
    def sites(self):
        if self.vector_direction == 'column':
            return self._sites
        else:
            return self._sites.T

    @property
    def vector_direction(self):
        return self._vector_direction

    @vector_direction.setter
    def vector_direction(self, vector_direction):
        vector_direction_setter(self, vector_direction)

    @staticmethod
    def concatenate(sites):
        """"""
        out = sites[0].copy()
        for i in sites[1:]:
            out += i

        return out

    def copy(self):
        """Make a copy of the Sites object."""
        return self.__copy__()

    def translate(self, vector):
        """Translate the coordinates by a vector.

        Parameters
        ----------
        vector : list of ndarray
            The vector must have the same dimension as the Sites object.

        Returns
        -------
        self

        """

        self.__iadd__(vector)

    def index(self, **kwargs):
        """Filter site indices by a label with a particular value."""

        match_label, match_val = self._validate_label_filter(**kwargs)
        label_vals = getattr(self, match_label)
        match_idx = np.where(label_vals == match_val)[0]

        return match_idx

    def whose(self, **kwargs):
        """Filter sites by a label with a particular value."""

        match_idx = self.index(**kwargs)
        match_sites = self._sites[:, match_idx]

        if self.vector_direction == 'row':
            match_sites = match_sites.T

        return match_sites

    def as_fractional(self, unit_cell):
        """Get sites in fractional units of a given unit cell.

        Parameters
        ----------
        unit_cell : ndarray of shape (3, 3)
            Either row or column vectors, depending on `vector_direction`.

        """

        if self.vector_direction == 'row':
            unit_cell = unit_cell.T

        unit_cell_inv = np.linalg.inv(unit_cell)
        sites_frac = np.dot(unit_cell_inv, self._sites)

        if self.vector_direction == 'row':
            sites_frac = sites_frac.T

        return sites_frac

    def get_plot_data(self, group_by=None):

        data = {
            'x': self._sites[0],
            'type': 'scatter',
            'mode': 'markers',
        }

        if self.dimension > 1:
            data.update({
                'y': self._sites[1],
            })

        if self.dimension > 2:
            data.update({
                'z': self._sites[2],
                'type': 'scatter3d',
            })

        return data
