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

    def __eq__(self, other):

        if self.__class__ != other.__class__:
            return False

        if self.name != other.name:
            return False

        if self.values.dtype.kind == 'f':  # floating point data
            if not np.allclose(self.values, other.values):
                return False
        else:
            if not np.all(self.values == other.values):
                return False

        return True

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
    """An ordered collection of points in N-dimensional space with arbitrary
    labelling.

    Attributes
    ----------
    coords : ndarray
    dimension : int
    vector_direction : str
    labels : dict of (str : (dict or SitesLabel))

    """

    # Prioritise our `__rmatmul__` over Numpy's `__matmul__`:
    __array_priority__ = 1

    __hash__ = None

    def __init__(self, coords, labels=None, vector_direction='column',
                 dimension=3, set_component_attributes=True):

        self.vector_direction = vector_direction
        self._coords = self._validate(coords, self.vector_direction, dimension)
        self._dimension = dimension
        self._set_component_attributes = set_component_attributes
        self._labels = self._init_labels(labels)

        self._single_sites = [SingleSite(sites=self, site_index=i)
                              for i in range(len(self))]

    def __setattr__(self, name, value):
        """Overridden method to prevent reassigning label attributes."""

        if getattr(self, '_labels', None) and name in self._labels:
            msg = 'Can\'t set attribute "{}"'.format(name)
            raise AttributeError(msg)

        # Set all other attributes as normal:
        super().__setattr__(name, value)

    def __len__(self):
        """Get how many coords there are in this Sites objects."""
        return self._coords.shape[1]

    def __getitem__(self, index):
        return self._single_sites[index]

    def __eq__(self, other):

        if self.__class__ == other.__class__:

            if not np.allclose(self._coords, other._coords):
                return False

            # Check for equal label keys:
            if list(set(self.labels.keys()).symmetric_difference(
                    set(other.labels.keys()))):
                return False

            # Check labels equal:
            for k, v in self.labels.items():
                if v != other.labels[k]:
                    return False

            return True

        else:
            return self.coords == other

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.coords < other.coords
        else:
            return self.coords < other

    def __gt__(self, other):
        if self.__class__ == other.__class__:
            return self.coords > other.coords
        else:
            return self.coords > other

    def __le__(self, other):
        if self.__class__ == other.__class__:
            return self.coords <= other.coords
        else:
            return self.coords <= other

    def __ge__(self, other):
        if self.__class__ == other.__class__:
            return self.coords >= other.coords
        else:
            return self.coords >= other

    def __copy__(self):
        out = Sites(
            coords=np.copy(self.coords),
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
            out._coords += self._validate_translation_vector(obj)

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

            new_sites = np.hstack([self._coords, obj._coords])
            self._coords = new_sites
            self._labels = new_labs

        else:
            # Add a translation vector:
            self._coords += self._validate_translation_vector(obj)

        return self

    def __sub__(self, vector):

        out = self.copy()
        out._coords -= self._validate_translation_vector(vector)

        return out

    def __rsub__(self, vector):

        out = self.copy()
        out._coords = self._validate_translation_vector(vector) - out._coords

        return out

    def __isub__(self, vector):

        self._coords -= self._validate_translation_vector(vector)

        return self

    def __mul__(self, number):
        """Scale coordinates by a scalar."""
        out = self.copy()
        out *= number
        return out

    def __rmul__(self, number):
        return self.__mul__(number)

    def __imul__(self, number):
        """Scale coordinates by a scalar."""
        if isinstance(number, numbers.Number):
            self._coords *= number
            return self

    def __truediv__(self, number):
        """Scale coordinates by a scalar."""
        out = self.copy()
        out /= number
        return out

    def __itruediv__(self, number):
        """Scale coordinates by a scalar."""
        if isinstance(number, numbers.Number):
            self._coords /= number
            return self

    def __matmul__(self, mat):
        """Transform site coordinates by a transformation matrix."""
        out = self.copy()
        out.__imatmul__(mat)
        return out

    def __rmatmul__(self, mat):
        """Transform site coordinates by a transformation matrix."""

        if self.vector_direction != 'column':
            msg = ('Cannot pre-multiply site coordinates by a transformation'
                   ' matrix when `Sites.vector_direction` is "row".')
            raise ValueError(msg)

        out = self.copy()
        out.transform(mat)
        return out

    def __imatmul__(self, mat):
        """Transform site coordinates by a transformation matrix."""

        if self.vector_direction != 'row':
            msg = ('Cannot post-multiply site coordinates by a transformation'
                   ' matrix when `Sites.vector_direction` is "column".')
            raise ValueError(msg)

        self.transform(mat)
        return self

    def _init_labels(self, labels):
        """Set labels as attributes for easy access."""

        BAD_LABELS = []
        if self.set_component_attributes:

            if self.dimension in [1, 2, 3]:
                BAD_LABELS = ['x']
                setattr(self, 'x', self.get_components(0))

            if self.dimension in [2, 3]:
                BAD_LABELS.append('y')
                setattr(self, 'y', self.get_components(1))

            if self.dimension in [3]:
                BAD_LABELS.append('z')
                setattr(self, 'z', self.get_components(2))

        label_objs = {}
        for k, v in (labels or {}).items():

            if k in BAD_LABELS:
                msg = ('Label name "{}" is a reserved component attribute '
                       'name. If you want to use this attribute to refer to '
                       'a label, instantiate the `Sites` object with '
                       '`set_component_attributes=False`.')
                raise ValueError(msg.format(k))

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

    def _validate(self, coords, vector_direction, dimension):
        """Validate inputs."""

        if dimension not in [2, 3]:
            msg = '`dimension` must be an integer: 2 or 3.'
            raise ValueError(msg)

        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)

        if coords.ndim != 2:
            raise ValueError('`coords` must be a 2D array.')

        vec_len_idx = 0 if vector_direction == 'column' else 1
        vec_len = coords.shape[vec_len_idx]

        if vec_len != dimension:
            msg = ('The length of {}s in `coords` ({}) must be equal to '
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
            return coords.T
        else:
            return coords

    def _validate_label_filter(self, **label_values):
        """Validation for the `index` method."""

        if not self.labels:
            raise ValueError(
                'No labels are associated with this Sites object.')

        if not label_values:
            msg = ('Provide a label condition to filter the sites. Available '
                   'labels are: {}')
            raise ValueError(msg.format(list(self.labels.keys())))

        if len(label_values) > 1:
            msg = 'Only one label condition is currently supported by `whose`.'
            raise NotImplementedError(msg)

        for match_label, match_val in label_values.items():
            try:
                getattr(self, match_label)
            except AttributeError as err:
                msg = 'No Sites label called "{}" was found.'
                raise ValueError(msg.format(match_label))

            return match_label, match_val

    def _validate_translation_vector(self, vector):
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

    def _validate_transformation_matrix(self, mat):
        """Try to validate the shape of the matrix, as intended to transform
        the site coordinates."""

        msg_all = 'Transformation matrix invalid: '

        if not isinstance(mat, np.ndarray):
            mat = np.array(mat)

        # must be 2D:
        if len(mat.shape) != 2:
            msg = msg_all + 'must be a 2D array.'
            raise ValueError(msg)

        # Assuming transformation does not change dimension, must be square:
        if mat.shape[0] != mat.shape[1]:
            msg = msg_all + ('must be a square matrix (dimension of '
                             'coordinates must not change).')
            raise ValueError(msg)

        # Axis size must be equal to dimension of coordinates:
        if mat.shape[0] != self.dimension:
            msg = msg_all + ('axis size must be equal to dimension of '
                             'coordinates ({})')
            raise ValueError(msg.format(self.dimension))

        return mat

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
    def set_component_attributes(self):
        return self._set_component_attributes

    @property
    def labels(self):
        return self._labels

    @property
    def dimension(self):
        return self._dimension

    @property
    def coords(self):
        if self.vector_direction == 'column':
            return self._coords
        else:
            return self._coords.T

    @property
    def vector_direction(self):
        return self._vector_direction

    @vector_direction.setter
    def vector_direction(self, vector_direction):
        vector_direction_setter(self, vector_direction)
        try:
            for i in self._single_sites:
                vector_direction_setter(i, vector_direction)
        except AttributeError:
            pass

    @property
    def centroid(self):
        """Get the geometric centre of the sites."""
        avg = np.mean(self._coords, axis=1)
        if self.vector_direction == 'column':
            avg = avg[:, None]
        return avg

    @property
    def bounding_box(self):
        """Get the orthogonal bounding "box" minima and maxima."""

        box = np.array([
            np.min(self._coords, axis=1),
            np.max(self._coords, axis=1)
        ])
        if self.vector_direction == 'column':
            box = box.T

        return box

    @staticmethod
    def concatenate(sites):
        """"""
        out = sites[0].copy()
        for i in sites[1:]:
            out += i

        return out

    @staticmethod
    def and_(*bool_arrs):
        """Convenience wrapper for Numpy's `logical_and`."""

        if not len(bool_arrs) > 1:
            msg = 'Pass at least two boolean arrays.'
            raise ValueError(msg)

        out = bool_arrs[0]
        for i in bool_arrs:
            out = np.logical_and(out, i)

        return out

    @staticmethod
    def or_(*bool_arrs):
        """Convenience wrapper for Numpy's `logical_or`."""

        if not len(bool_arrs) > 1:
            msg = 'Pass at least two boolean arrays.'
            raise ValueError(msg)

        out = bool_arrs[0]
        for i in bool_arrs:
            out = np.logical_or(out, i)

        return out

    def any(self, bool_arr):
        """Get 1-dimensional boolean array representing site indices where any
        components match an input boolean array.

        Parameters
        ----------
        bool_arr : ndarray of bool of shape equal to that of self.coords

        Returns
        -------
        ndarray of bool of shape (len(self), )

        """

        if bool_arr.shape != self.coords.shape:
            msg = ('`bool_arr` must have the same shape as the `coords` '
                   'attribute, which is {}.')
            raise ValueError(msg.format(self.coords.shape))
        axis = 0 if self.vector_direction == 'column' else 1

        return np.any(bool_arr, axis=axis)

    def all(self, bool_arr):
        """Get 1-dimensional boolean array representing site indices where all
        components match an input boolean array.

        Parameters
        ----------
        bool_arr : ndarray of bool of shape equal to that of self.coords

        Returns
        -------
        ndarray of bool of shape (len(self), )

        """

        if bool_arr.shape != self.coords.shape:
            msg = ('`bool_arr` must have the same shape as the `coords` '
                   'attribute, which is {}.')
            raise ValueError(msg.format(self.coords.shape))
        axis = 0 if self.vector_direction == 'column' else 1

        return np.all(bool_arr, axis=axis)

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

    def index(self, bool_arr=None, **label_values):
        """Filter site indices by a bool array or a label with a particular
        value.

        Parameters
        ----------
        bool_arr : ndarray of bool of shape (len(self),), optional
            If specified, get the indices (of sites) where bool_arr is True.
        label_values : dict
            label name and value to match

        Returns
        -------
        match_idx : ndarray of int
            Indices of sites that match the given condition (either a bool
            array or a particular label value).

        """

        if bool_arr is not None:
            if bool_arr.shape != (len(self),):
                msg = ('`bool_arr` must be a 1D array of length equal to the '
                       'number of sites, which is {}.')
                raise ValueError(msg.format(len(self)))
            condition = bool_arr

        else:
            match_label, match_val = self._validate_label_filter(
                **label_values)
            label_vals = getattr(self, match_label)
            condition = label_vals == match_val

        match_idx = np.where(condition)[0]

        return match_idx

    def where(self, bool_arr):
        """Filter sites by a bool array."""

        match_idx = self.index(bool_arr)
        match_sites = self._coords[:, match_idx]

        if self.vector_direction == 'row':
            match_sites = match_sites.T

        return match_sites

    def whose(self, **label_values):
        """Filter sites by a label with a particular value."""

        match_idx = self.index(**label_values)
        match_sites = self._coords[:, match_idx]

        if self.vector_direction == 'row':
            match_sites = match_sites.T

        return match_sites

    def as_fractional(self, unit_cell):
        """Get coords in fractional units of a given unit cell.

        Parameters
        ----------
        unit_cell : ndarray of shape (3, 3)
            Either row or column vectors, depending on `vector_direction`.

        """

        if self.vector_direction == 'row':
            unit_cell = unit_cell.T

        unit_cell_inv = np.linalg.inv(unit_cell)
        sites_frac = np.dot(unit_cell_inv, self._coords)

        if self.vector_direction == 'row':
            sites_frac = sites_frac.T

        return sites_frac

    def get_plot_data(self, group_by=None):

        data = {
            'x': self._coords[0],
            'type': 'scatter',
            'mode': 'markers',
        }

        if self.dimension > 1:
            data.update({
                'y': self._coords[1],
            })

        if self.dimension > 2:
            data.update({
                'z': self._coords[2],
                'type': 'scatter3d',
            })

        if self.dimension > 3:
            raise NotImplementedError

        return data

    def rotate(self, mat, centre=None):
        """Rotate the coordinates.

        Parameters
        ----------
        mat : ndarray 
            Rotation matrix to apply to the coordinates.
        centre : ndarray of size 3, optional
            Centre of rotation. If not specified, the Cartesian origin is used.

        """

        if not centre:
            centre = [0, 0, 0]

        centre = self._validate_translation_vector(
            centre)  # TODO rename this method

        self.translate(-centre)
        self.transform(mat)
        self.translate(centre)

    def transform(self, mat):

        mat = self._validate_transformation_matrix(mat)
        if self.vector_direction == 'row':
            mat = mat.T

        self._coords = mat @ self._coords

    def get_components(self, component_index):
        if component_index > (self.dimension - 1):
            msg = ('`Sites` object has dimension {} and so the maximum '
                   'component index is {}.')
            raise IndexError(msg.format(self.dimension, self.dimension - 1))
        return self._coords[component_index]


class SingleSite(Sites):
    """A single, labelled point in space."""

    def __init__(self, sites, site_index):

        self.sites = sites
        self.site_index = site_index

        self._coords = sites._coords[:, site_index][:, None]
        self._dimension = sites.dimension
        self._labels = self._init_labels()
        self._vector_direction = sites.vector_direction

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def _init_labels(self):
        """Set labels as attributes for easy access."""

        if self.sites.set_component_attributes:

            if self.dimension in [1, 2, 3]:
                setattr(self, 'x', self.get_components(0)[0])

            if self.dimension in [2, 3]:
                setattr(self, 'y', self.get_components(1)[0])

            if self.dimension in [3]:
                setattr(self, 'z', self.get_components(2)[0])

        labels = {}
        for k, v in self.sites._labels.items():

            val = v.values[self.site_index]
            sites_label = SitesLabel(
                k,
                values=np.array(val),
            )
            labels.update({
                k: sites_label
            })
            setattr(self, k, val)

        return labels

    def whose(self, **label_values):
        raise NotImplementedError

    def index(self, **label_values):
        raise NotImplementedError
