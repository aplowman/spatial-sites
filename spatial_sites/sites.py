"""`spatial_sites.sites.py`

Module defining a Sites class that represents a set of points in space.

"""

import numbers
import warnings
import re
import copy

import numpy as np

from spatial_sites.utils import check_indices, cast_arr_to_float, repr_dict

REPR_INDENT = 4


def refresh_visual(func):
    def func_wrapper(*args, **kwargs):
        # print('in decorator due to func: {}!'.format(func))
        out = func(*args, **kwargs)
        sites_obj = args[0]
        if isinstance(sites_obj, FilteredSites):
            sites_obj = sites_obj.sites
        for i in sites_obj.parent_visual_handlers:
            i(sites_obj)
        return out
    return func_wrapper


def vector_direction_setter(obj, vector_direction, warn=True):
    """Set the `vector_direction` attribute of a given object.

    Parameters
    ----------
    warn : bool, optional
        If True, a warning will be produced when the current value of the
        vector direction is already equivalent to the new value.

    """

    if vector_direction not in ['row', 'r', 'column', 'col', 'c']:
        msg = ('`vector_direction` must be specified as a string, either '
               '"row" (or "r") or "column" (or "col" or "c").')
        raise ValueError(msg)

    if vector_direction in ['col', 'c']:
        vector_direction = 'column'
    elif vector_direction == 'r':
        vector_direction = 'row'

    if warn:
        old_vec_dir = getattr(obj, '_vector_direction', None)
        if old_vec_dir:
            if vector_direction == old_vec_dir:
                msg = '`vector_direction` is already set to "{}"'
                warnings.warn(msg.format(vector_direction))

    obj._vector_direction = vector_direction


class Labels(object):
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

        if values is None:
            values = unique_values[values_idx]

        uniq_val, val_idx, counts = np.unique(values, return_inverse=True,
                                              return_counts=True)

        if unique_values is not None:

            # Check user-supplied unique values and idx make sense:
            if not np.all(np.sort(uniq_val) == np.sort(unique_values)):
                msg = ('Not all of the values in `unique_values` are unique.')
                raise ValueError(msg)

            # Check all `values_idx` do index `unique_values`:
            check_indices(unique_values, values_idx)

        self._validate_name(name)
        self.name = name
        self.unique_values = uniq_val
        self.values_idx = val_idx
        self.values_count = counts

    @property
    def values(self):
        return self.unique_values[self.values_idx]

    @property
    def dtype(self):
        return self.unique_values.dtype

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT
        out = (
            '{0}(\n'
            '{1}name={2!r},\n'
            '{1}unique_values={3!r},\n'
            '{1}values_idx={4!r},\n'
            '{1}values_count={5!r},\n'
            ')'.format(
                self.__class__.__name__,
                arg_fmt,
                self.name,
                self.unique_values,
                self.values_idx,
                self.values_count,
            )
        )

        return out

    def __str__(self):
        return '{}: {}'.format(self.name, self.values)

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


    def _validate_name(self, name):
        """Ensure name is safe to use as an object attribute."""
        pattern = r'^(?![0-9])[a-zA-Z0-9_]+$'
        if not re.match(pattern, name):
            msg = ('SitesLabel name "{}" is not valid since it cannot be '
                   'used as an object attribute name. Names must match the '
                   'regular expression "{}".')
            raise ValueError(msg.format(name, pattern))

    def remove(self, indices):
        """Remove multiple site labels according to an array of indices."""

        # Remove unwanted values:
        keep = np.ones(len(self), dtype=bool)
        keep[indices] = False
        values = self.values[keep]

        # Recompute unique values:
        uniq_val, val_idx, counts = np.unique(values, return_inverse=True,
                                              return_counts=True)
        self.unique_values = uniq_val
        self.values_idx = val_idx
        self.values_count = counts


class FilteredLabels(Labels):

    def __init__(self, labels_obj, keep_arr):

        self.name = labels_obj.name
        self.unique_values = labels_obj.unique_values
        self.values_idx = np.ma.array(labels_obj.values_idx, mask=~keep_arr)
        self.values_count = labels_obj.values_count

    @property
    def values(self):
        return self.unique_values[self.values_idx.compressed()]


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
                 dimension=3, component_labels=None, basis=None,
                 cast_to_float=False):
        """
        Parameters
        ----------
        component_labels : list of str or False, optional
            If specified, must be a list (of strings) of length equal to
            `dimension`. Coordinate component will then be assigned to
            instance attributes with these names. If False, no component
            attributes will be set. By default, set to `None`, in which case
            labels "x", "y", and "z" will be used (as appropriate, given
            `dimension`).
        cast_to_float : bool, optional
            If True and the dtype of `coords` is not a floating point dtype,
            `coords`, elements of `coords` will be cast to a floating point
            numbers using the fractions module.

        """

        self.vector_direction = vector_direction
        self._coords = self._validate(coords, self.vector_direction, dimension,
                                      cast_to_float)
        self._dimension = dimension
        self.basis = basis

        self._bad_label_names = self._get_bad_label_names()
        self._component_labels = self._get_component_labels(component_labels)
        self._set_component_attrs()

        self._labels = self._init_labels(labels)

        self._single_sites = [SingleSite(sites=self, site_index=i)
                              for i in range(len(self))]

        self.parent_visual_handlers = []

    def __setattr__(self, name, value):
        """Overridden method to prevent reassigning label and component
        attributes."""

        if getattr(self, '_labels', None) and name in self._labels:
            msg = 'Cannot set attribute "{}"'.format(name)
            raise AttributeError(msg)

        if getattr(self, '_component_labels', None):
            if name in self.component_labels:
                msg = 'Cannot set attribute "{}"'.format(name)
                raise AttributeError(msg)

        # Set all other attributes as normal:
        super().__setattr__(name, value)

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT

        coords = '{!r}'.format(self.coords)
        coords = coords.replace('\n', '\n' + arg_fmt + ' ' * len('coords='))
        labels = repr_dict(self.labels, REPR_INDENT)

        out = (
            '{0}(\n'
            '{1}dimension={2!r},\n'
            '{1}vector_direction={3!r},\n'
            '{1}component_labels={4!r},\n'
            '{1}coords={5},\n'
            '{1}labels={6},\n'
            ')'.format(
                self.__class__.__name__,
                arg_fmt,
                self.dimension,
                self.vector_direction,
                self.component_labels,
                coords,
                labels,
            )
        )
        return out

    def __str__(self):

        labels = []
        for k, v in self.labels.items():
            labels.append('{!s}'.format(v))

        out = '{}\n\n{}\n'.format(self.coords, '\n'.join(labels))
        return out

    def __len__(self):
        """Get how many coords there are in this Sites objects."""
        return self.__coords.shape[1]

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            return self._single_sites[index]
        elif isinstance(index, (list, np.ndarray)):
            # "Fancy indexing" (1D only)
            index = np.array(index)
            if index.ndim > 1:
                msg = ('Index array must have one dimension, but has {} dimensions.')
                raise IndexError(msg.format(index.ndim))

            new_coords = self._coords[:, index]
            if self.vector_direction == 'row':
                new_coords = new_coords.T

            new_labs = {}
            for lab_name, lab in self.labels.items():
                new_labs.update({lab_name: Labels(lab_name, lab.values[index])})

            return Sites(
                coords=new_coords,
                labels=new_labs,
                vector_direction=self.vector_direction,
                dimension=self.dimension,
                component_labels=self.component_labels,
                basis=self.basis,
            )

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

                sites_lab_new = Labels(name=lab_name, values=new_lab_vals)
                new_labs.update({
                    lab_name: sites_lab_new,
                })
                super().__setattr__(lab_name, sites_lab_new.values)

            new_sites = np.hstack([self._coords, obj._coords])
            self._labels = new_labs
            self._coords = new_sites

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

    def __mul__(self, multiplier):
        'Multiplication by scalar or broadcastable Numpy array.'
        out = self.copy()
        out *= multiplier
        return out

    def __rmul__(self, number):
        return self.__mul__(number)

    def __imul__(self, multiplier):
        'In-place multiplication by scalar or broadcastable Numpy array.'
        multiplier = np.array(multiplier)
        if self.vector_direction == 'row':
            multiplier = multiplier.T
        self._coords *= multiplier
        return self

    def __truediv__(self, divisor):
        'True division by scalar or broadcastable Numpy array.'
        out = self.copy()
        out /= divisor
        return out

    def __itruediv__(self, divisor):
        'In-place true division by scalar or broadcastable Numpy array.'
        divisor = np.array(divisor)
        if self.vector_direction == 'row':
            divisor = divisor.T
        self._coords /= divisor
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

    def _get_component_labels(self, component_labels):

        out = []

        if component_labels is None:
            if self.dimension in [1, 2, 3]:
                out.append('x')

            if self.dimension in [2, 3]:
                out.append('y')

            if self.dimension == 3:
                out.append('z')

        elif component_labels:
            if len(component_labels) != self.dimension:
                msg = ('If specifying `component_labels`, the list must be the'
                       ' same length as the number of dimensions, but "{}" was '
                       'specified.')
                raise ValueError(msg.format(component_labels))

            out = component_labels
            for i in component_labels:
                if i in self._bad_label_names:
                    msg = '"{}" cannot be used as a component attribute name.'
                    raise ValueError(msg.format(i))

        self._bad_label_names += out

        return out

    def _get_coords(self, new_basis):

        try:
            old_basis = self._basis
        except AttributeError:
            old_basis = None

        if old_basis is not None:

            try:
                new_basis_inv = np.linalg.inv(new_basis)
            except np.linalg.LinAlgError:
                msg = ('New basis matrix is singular and so does not '
                       'represent a basis set.')
                raise ValueError(msg)

            # Transform from old basis to standard, then from standard to new:
            coords = new_basis_inv @ old_basis @ self._coords

        else:
            # If no existing basis, coords are already in the correct basis:
            coords = self._coords

        return coords

    def _get_bad_label_names(self):

        bad_labels = [
            'bad_label_names',
            'vector_direction',
            'coords',
            'dimension',
            'component_labels',
            'labels',
            'single_sites',
        ]

        # Include "underscored" versions of attributes names:
        bad_labels = [j for i in bad_labels for j in [i, '_' + i]]

        return bad_labels

    def _set_component_attrs(self):
        """Called on instantiation to set coordinate attributes like e.g. `x`
        to the first coordinates component."""

        if hasattr(self, '_component_labels') and self._component_labels:
            for i in range(self.dimension):
                if self._component_labels[i]:
                    super().__setattr__(self._component_labels[i], self.get_components(i))

    def _init_labels(self, labels):
        """Set labels as attributes for easy access."""

        label_objs = {}
        for k, v in (labels or {}).items():

            if k in self._bad_label_names:
                msg = 'Label name "{}" is a reserved attribute name.'
                raise ValueError(msg.format(k))

            if isinstance(v, Labels):
                sites_label = v

            else:
                values = None
                unique_values = None
                values_idx = None

                if len(v) == 2 and isinstance(v[0], (np.ndarray, list, tuple)):
                    unique_values, values_idx = v
                else:
                    values = v

                sites_label = Labels(
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

    def _validate(self, coords, vector_direction, dimension, cast_to_float):
        """Validate inputs."""

        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)

        if cast_to_float:
            coords = cast_arr_to_float(coords)

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

        label_vals = {}
        for match_label, match_val in label_values.items():
            try:
                getattr(self, match_label)
            except AttributeError:
                msg = 'No Sites label called "{}" was found.'
                raise ValueError(msg.format(match_label))

            label_vals.update({
                match_label: match_val,
            })

        return label_vals

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

        if vector.shape == self.coords.shape:
            return vector
        else:
            if len(vector.shape) > 1:
                vector = np.squeeze(vector)
            if vector.shape == (self.dimension, ):
                return vector[:, None]
            else:
                msg = ('Cannot translate coordinates with shape {} by an '
                       'array with shape {}.')
                raise ValueError(msg.format(self.coords.shape, vector.shape))

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

    def _validate_new_basis(self, new_basis):

        dim = self.dimension
        req_shape = (dim, dim)

        if new_basis is None:
            # Set the default basis to the standard basis:
            new_basis = np.eye(dim)

        if not isinstance(new_basis, np.ndarray):
            new_basis = np.array(new_basis)

        if new_basis.shape != req_shape:
            msg = '`new_basis` must be an array with shape {}.'
            raise ValueError(msg.format(req_shape))

        if self.vector_direction == 'row':
            # Must use matrices of column vectors for both old and new bases:
            new_basis = new_basis.T

        return new_basis

    @property
    def component_labels(self):
        return self._component_labels

    @property
    def labels(self):
        return self._labels

    @property
    def dimension(self):
        return self._dimension

    @property
    def _coords(self):
        return self.__coords

    @_coords.setter
    def _coords(self, _coords):

        self.__coords = _coords
        self._set_component_attrs()
        try:
            self._single_sites = [SingleSite(sites=self, site_index=i)
                                  for i in range(len(self))]
        except AttributeError:
            pass

    @property
    def coords(self):
        if self.vector_direction == 'column':
            return self._coords
        else:
            return self._coords.T

    @property
    def basis(self):
        if self.vector_direction == 'column':
            return self._basis
        else:
            return self._basis.T

    @basis.setter
    def basis(self, new_basis):
        """Set or change the basis of the coordinates."""

        new_basis = self._validate_new_basis(new_basis)
        self._coords = self._get_coords(new_basis)
        self._basis = new_basis

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
        return copy.deepcopy(self)

    @refresh_visual
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

    def index_arr(self, bool_arr=None, inverse=False, label_match_mode='and',
                  **label_values):
        """Get the indices of sites using a bool array or labels with particular values.

        Parameters
        ----------
        bool_arr : ndarray of bool of shape (len(self),), optional
            If specified, get the indices (of sites) where bool_arr is True. Specify
            exactly one of `bool_arr` and `label_vales`.        
        inverse : bool
            If True, get the indices when either the bool array is False, or where the
            labels do not have the specified values. For a site index to be returned, all
            of the specified labels must not match the given values. By default, False.
        label_match_mode : str
            Determines whether to combine label value conditions using logical AND ("and")
            (default) or logical OR ("or").
        label_values : dict
            label names and values to match. Specify exactly one of `bool_arr` and
            `label_vales`. For a site index to be returned, all of the specified labels
            values must match the given values.

        Returns
        -------
        match_idx : ndarray of int
            Indices of sites that match the given condition (either a bool array or a
            particular label value).

        """

        if bool_arr is not None:
            if bool_arr.shape != (len(self),):
                msg = ('`bool_arr` must be a 1D array of length equal to the '
                       'number of sites, which is {}.')
                raise ValueError(msg.format(len(self)))
            condition = np.copy(bool_arr)
            if inverse:
                condition = np.logical_not(condition)
        else:
            label_match_vals = self._validate_label_filter(**label_values)
            condition = None
            for match_label, match_val in label_match_vals.items():
                label_vals = getattr(self, match_label)
                if inverse:
                    condition_i = label_vals != match_val
                else:
                    condition_i = label_vals == match_val
                if condition is None:
                    condition = condition_i
                else:
                    if label_match_mode == 'and':
                        condition = np.logical_and(condition, condition_i)
                    elif label_match_mode == 'or':
                        condition = np.logical_or(condition, condition_i)

        return condition

    def index(self, bool_arr=None, inverse=False, label_match_mode='and', **label_values):
        match_arr = self.index_arr(bool_arr, inverse, label_match_mode, **label_values)
        match_idx = np.where(match_arr)[0]
        return match_idx

    def where(self, bool_arr):
        """Get coordinates indexed by a bool array."""

        match_idx = self.index(bool_arr)
        match_sites = self._coords[:, match_idx]

        if self.vector_direction == 'row':
            match_sites = match_sites.T

        return match_sites

    def whose(self, inverse=False, label_match_mode='and', **label_values):
        """Get coordinates whose labels have a particular value."""

        match_idx = self.index(None, inverse, label_match_mode, **label_values)
        match_sites = self._coords[:, match_idx]

        if self.vector_direction == 'row':
            match_sites = match_sites.T

        return match_sites

    def filter(self, bool_arr=None, copy=False, label_match_mode='and', **label_values):
        """Filter using a bool array or label values.

        Parameters
        ----------
        bool_arr : ndarray of bool of shape (len(self),), optional
            If specified, filter sites by the indices where bool_arr is True. Specify
            exactly one of `bool_arr` and `label_vales`.        
        copy : bool, optional
            If True, return a new `Sites` object, with only coordinates matching the
            filtering conditions. If False, return a `FilteredSites` object, that acts
            as a view into this `Sites` object. False by default.
        label_values : dict
            label names and values to match. Specify exactly one of `bool_arr` and
            `label_vales`.

        Returns
        -------
        filtered_sites : Sites or FilteredSites            

        """
        if copy:
            filtered_sites = self.copy()
            filtered_sites.remove(bool_arr, True, label_match_mode, **label_values)
        else:
            filtered_sites = FilteredSites(
                sites_obj=self,
                bool_arr=bool_arr,
                label_match_mode=label_match_mode,
                **label_values
            )

        return filtered_sites

    def remove_idx(self, match_idx):
        'Remove sites based on an index array.'
        keep = np.ones(len(self), dtype=bool)
        keep[match_idx] = False

        for label_name, sites_label in self.labels.items():
            sites_label.remove(match_idx)
            super().__setattr__(label_name, sites_label.values)

        self._coords = self._coords[:, keep]
        # self._single_sites = [i for i, j in zip(self._single_sites, keep) if j]

    def remove(self, bool_arr=None, inverse=False, label_match_mode='and', **label_values):
        'Remove sites based on a bool_arr or a label value.'
        self.remove_idx(self.index(bool_arr, inverse, label_match_mode, **label_values))

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

        if centre is None:
            centre = [0, 0, 0]

        centre = self._validate_translation_vector(centre)  # TODO rename this method

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

    def add_labels(self, **labels):
        """Associate more labels with the coordinates."""

        for label_name in labels:
            if getattr(self, '_labels', None):
                if label_name in self.labels:
                    msg = ('Cannot add a new label named "{}"; it already '
                           'exists.')
                    raise ValueError(msg.format(label_name))

        new_labels = self._init_labels(labels)
        self._labels.update(new_labels)

        try:
            for i in self._single_sites:
                i._labels.update(i._init_labels(new_labels))
        except AttributeError:
            pass

    def remove_labels(self, *label_names):
        """Remove some of the labels associated with the coordinates."""

        for i in label_names:
            if i not in self.labels:
                msg = 'Cannot remove label named "{}"; it does not exist.'
                raise ValueError(msg.format(i))

            # Remove from labels dict:
            self._labels.pop(i)

            # Remove attribute:
            delattr(self, i)

            # Remove label from `SingleSite`s:
            for j in self._single_sites:
                j._labels.pop(i)
                delattr(j, i)

    def get_coords(self, new_basis=None):
        """Get coordinates in another basis. By default, coordinates are
        returned in the standard basis."""

        new_basis = self._validate_new_basis(new_basis)
        coords = self._get_coords(new_basis)

        if self.vector_direction == 'row':
            coords = coords.T

        return coords

    def tile(self, other_sites, repeat_labels=None):
        """For each coordinate, generate more coordinates by combining
        coordinates with another Sites object (via addition).

        Parameters
        ----------
        other_sites : Sites
        repeat_labels : dict
            Keys are string label names, values are label names to add to the
            new Sites object that enable the original order of repeated labels
            values to be identified.

        Returns
        -------
        tiled_sites

        """

        if not repeat_labels:
            repeat_labels = {}

        if not isinstance(other_sites, self.__class__):
            msg = 'Pass another Sites object with which to tile.'
            raise ValueError(msg)

        if not self.dimension == other_sites.dimension:
            msg = 'Both Sites must share the same dimension.'
            raise ValueError(msg)

        rep_labs = list(self.labels.keys() & other_sites.labels.keys())
        if rep_labs:
            msg = 'Cannot tile Sites that share the same label names: {}'
            raise ValueError(msg.format(rep_labs))

        # Merge coordinates
        base_coords = self._coords.T.reshape((-1, self.dimension, 1))
        tile_coords = other_sites._coords
        new_coords = np.hstack(base_coords + tile_coords)

        # Merge labels
        new_labels = {}
        for label_name, label in self.labels.items():

            if label_name in repeat_labels:
                uniq_lab_order = np.zeros_like(label.values_idx)
                for count_idx, i in enumerate(label.values_count):
                    uniq_lab_order[label.values_idx == count_idx] = range(i)

                uniq_lab_order = np.repeat(uniq_lab_order, len(other_sites))
                new_labels.update({
                    repeat_labels[label_name]: Labels(
                        name=repeat_labels[label_name],
                        values=uniq_lab_order,
                    )
                })

            new_labels.update({
                label_name: np.repeat(label.values, len(other_sites))
            })

        for label_name, label in other_sites.labels.items():

            if label_name in repeat_labels:
                uniq_lab_order = np.zeros_like(label.values_idx)
                for count_idx, i in enumerate(label.values_count):
                    uniq_lab_order[label.values_idx == count_idx] = range(i)

                uniq_lab_order = np.tile(uniq_lab_order, len(self))
                new_labels.update({
                    repeat_labels[label_name]: Labels(
                        name=repeat_labels[label_name],
                        values=uniq_lab_order,
                    )
                })

            new_labels.update({
                label_name: np.tile(label.values, len(self))
            })

        if self.vector_direction == 'row':
            new_coords = new_coords.T

        if self.vector_direction != other_sites.vector_direction:
            msg = ('`vector_direction`s are not the same for both Sites '
                   'objects. The `vector_direction` for the current Sites '
                   'object ({}) will be used.')
            warnings.warn(msg.format(self.vector_direction))

        if self.component_labels != other_sites.component_labels:
            msg = ('`component_labels`s are not the same for both Sites '
                   'objects. The `component_labels` for the current Sites '
                   'object ({}) will be used.')
            warnings.warn(msg.format(self.component_labels))

        out = Sites(
            coords=new_coords,
            labels=new_labels,
            dimension=self.dimension,
            vector_direction=self.vector_direction,
            component_labels=self.component_labels,
            basis=self.basis,
        )

        return out

    def to_homogeneous(self, component_label='w'):
        'Add another dimension with components 1.'
        self._coords = np.vstack([self._coords, np.ones(len(self))])
        self._dimension += 1
        self._basis = np.vstack([
            np.hstack([self.basis, np.zeros((3, 1))]),
            np.array([0, 0, 0, 1])
        ])

        for idx, i in enumerate(self._single_sites):
            i._dimension = self.dimension
            i._coords = self._coords[:, idx][:, None]

        if component_label:
            if hasattr(self, component_label):
                msg = ('Cannot label homogeneous coordinate as "{}", since this is '
                       'already an attribute.'.format(component_label))
                raise ValueError(msg)
            else:
                self._component_labels.append(component_label)
                self._bad_label_names.append(component_label)
                super().__setattr__(component_label,
                                    self.get_components(self.dimension - 1))

                for i in self._single_sites:
                    super(Sites, i).__setattr__(component_label,
                                                i.get_components(self.dimension - 1))

        return self

    def from_homogeneous(self):
        'Divide by and then remove the final component.'

        comp_label = None
        if len(self.component_labels) == self.dimension:
            comp_label = self._component_labels.pop()
            self._bad_label_names.remove(comp_label)
            self.__delattr__(comp_label)

        self._coords /= self._coords[-1]  # "Perspective divide"
        self._coords = self._coords[:-1]
        self._dimension -= 1
        self._basis = self._basis[0:-1, 0:-1]

        for idx, i in enumerate(self._single_sites):
            i._dimension = self.dimension
            i._coords = self._coords[:, idx][:, None]
            if comp_label:
                i.__delattr__(comp_label)

        return self

    def snap_coords(self, vals, tol=1e-14):
        'Snap coordinates values'

        for i in vals:
            self._coords[abs(self._coords - i) < tol] = i


class SingleSite(Sites):
    """A single, labelled point in space."""

    def __init__(self, sites, site_index):

        self.sites = sites
        self.site_index = site_index

        self._coords = sites._coords[:, site_index][:, None]
        self._dimension = sites._dimension
        self._component_labels = sites._component_labels
        self._set_component_attrs()
        self._labels = self._init_labels(sites._labels)
        self._vector_direction = sites.vector_direction

    @property
    def _coords(self):
        return self.__coords

    @_coords.setter
    def _coords(self, _coords):
        self.__coords = _coords

    def __repr__(self):

        arg_fmt = ' ' * REPR_INDENT

        sites = '{!r}'.format(self.sites)
        sites = sites.replace('\n', '\n' + arg_fmt)

        out = (
            '{0}(\n'
            '{1}site_index={2!r},\n'
            '{1}sites={3},\n'
            ')'.format(
                self.__class__.__name__,
                arg_fmt,
                self.site_index,
                sites,
            )
        )
        return out

    def __str__(self):

        labels = []
        for k, v in self.labels.items():
            labels.append('{}: {}'.format(k, v.values[0]))

        return '{}\n\n{}\n'.format(self.coords, '\n'.join(labels))

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def _set_component_attrs(self):
        """Called on instantiation to set coordinate attributes like e.g. `x`
        to the first coordinates component."""

        if self._component_labels:
            for i in range(self.dimension):
                if self._component_labels[i]:
                    super(Sites, self).__setattr__(
                        self._component_labels[i],
                        self.get_components(i)
                    )

    def _init_labels(self, labels):
        """Set labels as attributes for easy access."""

        label_objs = {}
        for k, v in labels.items():

            val = v.values[self.site_index]
            sites_label = Labels(
                k,
                values=np.array(val),
            )
            label_objs.update({
                k: sites_label
            })
            setattr(self, k, val)

        return label_objs

    def get_components(self, component_index):
        return super().get_components(component_index)[0]

    def index(self, **label_values):
        raise NotImplementedError

    def where(self, bool_arr):
        raise NotImplementedError

    def whose(self, **label_values):
        raise NotImplementedError

    def remove(self, bool_arr=None, **label_values):
        raise NotImplementedError

    def add_labels(self, **labels):
        raise NotImplementedError

    def remove_labels(self, *label_names):
        raise NotImplementedError


class FilteredSites(Sites):
    'A view into a Sites object where some coordinates are masked.'

    def __init__(self, sites_obj, bool_arr=None, **label_values):
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

        keep_arr = sites_obj.index_arr(bool_arr, **label_values)
        keep_arr_tiled = np.tile(keep_arr, (sites_obj.dimension, 1))
        self.sites = sites_obj
        self._keep_arr = keep_arr

        self.vector_direction = sites_obj.vector_direction
        self._coords = np.ma.array(sites_obj._coords, mask=~keep_arr_tiled)
        self._dimension = sites_obj.dimension
        self.basis = sites_obj.basis

        self._bad_label_names = sites_obj._bad_label_names
        self._component_labels = sites_obj._component_labels

        # self._set_component_attrs() # TODO

        self._labels = {k: FilteredLabels(v, keep_arr)
                        for k, v in sites_obj._labels.items()}

        for k, v in self._labels.items():
            super(Sites, self).__setattr__(k, v.values)

        self._single_sites = [i for idx, i in enumerate(sites_obj._single_sites)
                              if keep_arr[idx]]

        self.parent_visual_handlers = sites_obj.parent_visual_handlers

    @property
    def _coords(self):
        return self.__coords

    @_coords.setter
    def _coords(self, _coords):
        self.__coords = _coords

    def transform(self, mat):

        if not np.any(self._keep_arr):
            return

        mat = self._validate_transformation_matrix(mat)
        if self.vector_direction == 'row':
            mat = mat.T

        operate_arr = np.logical_not(np.all(self._coords.mask, axis=0))

        self._coords[:, operate_arr] = mat @ self._coords[:, operate_arr].data

    def translate(self, vector):

        if not np.any(self._keep_arr):
            return

        super().translate(vector)
