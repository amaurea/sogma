# This module contains the tiny subset of sotodlib functionality
# sogma uses. It's useful to avoid problems with the hard-to-compile
# so3g and spt3g modules that sotodlib pulls in, but sogma doesn't
# actually use.

# What we need from sotodlib:
# * core.metadata.resultset.ResultSet ok
# * core.LabelAxis ok
# * core.OffsetAxis ok
# * core.AxisManager ok
# * core.metadata.ObsFileDb ok
#
# This could be stubbed much further than what I do here, but this
# should be enough for now

# These are copied from sotodlib, with some unneeded stuff removed
import numpy as np, warnings, sqlite3, os, sys
from collections import OrderedDict

class AxisInterface:
    """Abstract base class for axes managed by AxisManager."""
    count = None
    name = None
    def __init__(self, name): self.name = name
    def __repr__(self): raise NotImplementedError
    def _minirepr_(self): return self.__repr__()
    def copy(self): raise NotImplementedError
    def rename(self, name): self.name = name
    def resolve(self, src, axis_index=None):
        """Perform a check or promote-and-check of this Axis against a data
        object.

        The promotion step only applies to "unset" Axes, i.e. those
        here count is None.  Not all Axis types will be able to
        support this.  Promotion involves inspection of src and
        axis_index to fix free parameters in the Axis.  If promotion
        is successful, a new ("set") Axis is returned.  If promotion
        is attempted and fails then a ValueError is raised.X

        The check step involes confirming that the data object
        described by src (and axis_index) is compatible with the
        current axis (or with the axis resulting from axis Promotion).
        Typically that simply involves confirming that
        src.shape[axis_index] == self.count.  If the check fails, a
        ValueError is raised.

        Arguments:
          src: The data object to be wrapped (e.g. a numpy array)
          axis_index: The index of the data object to test for
            compatibility.

        Returns:
          axis: Either self, or the result of promotion.

        """
        # The default implementation performs the check step.
        # Subclasses may attempt promotion, then call this.
        ref_count = src.shape[axis_index]
        if self.count != ref_count:
            raise ValueError(
                "Dimension %i of data is incompatible with axis %s" %
                (axis_index, repr(self)))
        return self
    def restriction(self, selector):
        """Apply `selector` to the elements of this axis, returning a new Axis
        of the same type and an array indexing object (a slice or an
        array of integer indices) that may be used to extract the
        corresponding elements from a vector.

        See class header for acceptable selector objects.

        Returns (new_axis, ar_index).

        """
        raise NotImplementedError
    def intersection(self, friend, return_slices=False):
        """Find the intersection of this Axis and the friend Axis, returning a
        new Axis of the same type.  Optionally, also return array
        indexing objects that select the common elements from array
        dimensions corresponding to self and friend, respectively.

        See class header for acceptable selector objects.

        Returns (new_axis), or (new_axis, ar_index_self,
        ar_index_friend) if return_slices is True.

        """
        raise NotImplementedError

class OffsetAxis(AxisInterface):
    """This class manages an integer-indexed axis, with an accounting for
    an integer offset of any single vector relative to some absolute
    reference point.  For example, one vector could could have 100
    elements at offset 50, and a second vector could have 100 elements
    at offset -20.  On intersection, the result would have 30 elements
    at offset 50.

    The property `origin_tag` may be used to identify the absolute
    reference point.  It could be a TOD name ('obs_2020-12-01') or a
    timestamp or whatever.

    Selectors must be slice objects (with stride 1!) or tuples to be
    passed into slice(), e.g. (0, 1000) or (0, None, 1).

    """

    origin_tag = None
    offset = 0

    def __init__(self, name, count=None, offset=0, origin_tag=None):
        super().__init__(name)
        self.count = count
        self.offset = offset
        self.origin_tag = origin_tag

    def copy(self): return OffsetAxis(self.name, self.count, self.offset, self.origin_tag)
    def __repr__(self): return 'OffsetAxis(%s:%s%+i)' % (self.count, self.origin_tag, self.offset)
    def _minirepr_(self): return 'OffsetAxis(%s)' % (self.count)
    def resolve(self, src, axis_index=None):
        if self.count is None:
            return OffsetAxis(self.name, src.shape[axis_index])
        return super().resolve(src, axis_index)
    def restriction(self, selector):
        if not isinstance(selector, slice):
            sl = slice(*selector)
        else:
            sl = selector
        start, stop, stride = sl.indices(self.count + self.offset)
        assert stride == 1
        assert start >= self.offset
        assert stop <= self.offset + self.count
        return (OffsetAxis(self.name, stop - start, start, self.origin_tag),
                slice(start - self.offset, stop - self.offset, stride))
    def __eq__(self, other):
        return (self.count == other.count and
                self.offset == other.offset and
                self.origin_tag == other.origin_tag)
    def intersection(self, friend, return_slices=False):
        offset = max(self.offset, friend.offset)
        count = min(self.count + self.offset,
                    friend.count + friend.offset) - offset
        count = max(count, 0)
        ax = OffsetAxis(self.name, count, offset, self.origin_tag)
        if return_slices:
            return ax, \
                slice(offset - self.offset, count + offset - self.offset), \
                slice(offset - friend.offset, count + offset - friend.offset)
        else:
            return ax

class LabelAxis(AxisInterface):
    """This class manages a string-labeled axis, i.e., an axis where each
    element has been given a unique name.  The vector of names can be
    found in self.vals.

    Instantiation with labels that are not strings will raise a TypeError.

    On intersection of two vectors, only elements whose names appear
    in both axes will be preserved.

    Selectors should be lists (or arrays) of label strings.

    """

    def __init__(self, name, vals=None):
        super().__init__(name)
        if vals is not None:
            if len(vals): vals = np.array(vals)
            else:         vals = np.array([], dtype=np.str_)
            if vals.dtype.type is not np.str_:
                raise TypeError('LabelAxis labels must be strings not %s' % vals.dtype)
        self.vals = vals

    @property
    def count(self):
        if self.vals is None:
            return None
        return len(self.vals)

    def __repr__(self):
        if self.vals is None:
            items = ['?']
        elif len(self.vals) > 20:
            items = ([repr(v) for v in self.vals[:3]] + ['...'] +
                     [repr(v) for v in self.vals[-4:]])
        else:
            items = [repr(v) for v in self.vals]
        return 'LabelAxis(%s:' % self.count + ','.join(items) + ')'

    def _minirepr_(self): return 'LabelAxis(%s)' % (self.count)
    def copy(self): return LabelAxis(self.name, self.vals)

    def resolve(self, src, axis_index=None):
        if self.count is None:
            raise RuntimeError(
                'LabelAxis cannot be naively promoted from data.')
        return super().resolve(src, axis_index)

    def restriction(self, selector):
        # Selector should be list of vals or a mask. Returns new axis and the
        # indices into self.vals that project out the elements.
        if self.vals is not None and isinstance(selector, np.ndarray) and selector.dtype == bool:
            selector = self.vals[selector]
        _, i0, i1 = get_coindices(selector, self.vals)
        assert len(i0) == len(selector)  # not a strict subset!
        return LabelAxis(self.name, selector), i1

    def __eq__(self, other): return (self.count == other.count and np.all(self.vals == other.vals))

    def intersection(self, friend, return_slices=False):
        _vals, i0, i1 = get_coindices(self.vals, friend.vals)
        ax = LabelAxis(self.name, _vals)
        if return_slices: return ax, i0, i1
        else: return ax

def get_coindices(v0, v1, check_unique=False):
    """Given vectors v0 and v1, each of which contains no duplicate
    values, determine the elements that are found in both vectors.
    Returns (vals, i0, i1), i.e. the vector of common elements and
    the vectors of indices into v0 and v1 where those elements are
    found.

    This routine will use np.intersect1d if it can.  The ordering of
    the results is different from intersect1d -- vals is not sorted,
    but rather the elements will appear in the same order that they
    were found in v0 (so that i0 is strictly increasing).

    The behavior is undefined if either v0 or v1 contain duplicates.
    Pass check_unique=True to assert that condition.

    """
    if check_unique:
        assert(len(set(v0)) == len(v0))
        assert(len(set(v1)) == len(v1))

    try:
        vals, i0, i1 = np.intersect1d(v0, v1, return_indices=True)
        order = np.argsort(i0)
        return vals[order], i0[order], i1[order]
    except TypeError:  # return_indices not implemented in numpy < 1.15
        pass

    # The old fashioned way
    v0 = np.asarray(v0)
    w0 = sorted([(j, i) for i, j in enumerate(v0)])
    w1 = sorted([(j, i) for i, j in enumerate(v1)])
    i0, i1 = 0, 0
    pairs = []
    while i0 < len(w0) and i1 < len(w1):
        if w0[i0][0] == w1[i1][0]:
            pairs.append((w0[i0][1], w1[i1][1]))
            i0 += 1
            i1 += 1
        elif w0[i0][0] < w1[i1][0]:
            i0 += 1
        else:
            i1 += 1
    if len(pairs) == 0:
        return (np.zeros(0, v0.dtype), np.zeros(0, int), np.zeros(0, int))
    pairs.sort()
    i0, i1 = np.transpose(pairs)
    return v0[i0], i0, i1

class AxisManager:
    """A container for numpy arrays and other multi-dimensional
    data-carrying objects (including other AxisManagers).  This object
    keeps track of which dimensions of each object are concordant, and
    allows one to slice all hosted data simultaneously.
    """
    def __init__(self, *args):
        self._axes = OrderedDict()
        self._assignments = {}  # data_name -> [ax0_name, ax1_name, ...]
        self._fields = OrderedDict()
        for a in args:
            if isinstance(a, AxisManager):
                # merge in the axes and copy in the values.
                self.merge(a)
            elif isinstance(a, AxisInterface):
                self._axes[a.name] = a.copy()
            else:
                raise ValueError("Cannot handle type %s in constructor." % a)

    @property
    def shape(self):
        return tuple([a.count for a in self._axes.values()])

    def copy(self, axes_only=False):
        out = AxisManager()
        for k, v in self._axes.items():
            out._axes[k] = v
        if axes_only:
            return out
        for k, v in self._fields.items():
            if np.isscalar(v) or v is None:
                out._fields[k] = v
            else:
                out._fields[k] = v.copy()
        for k, v in self._assignments.items():
            out._assignments[k] = v.copy()
        return out

    def _managed_ids(self):
        ids = [id(self)]
        for v in self._fields.values():
            if isinstance(v, AxisManager):
                ids.extend(v._managed_ids())
        return ids

    def __delitem__(self, name):
        if name in self._fields:
            del self._fields[name]
            del self._assignments[name]
        elif name in self._axes:
            del self._axes[name]
            for v in self._assignments.values():
                for i, n in enumerate(v):
                    if n == name:
                        v[i] = None
        else:
            raise KeyError(name)

    def move(self, name, new_name):
        """Rename or remove a data field.  To delete the field, pass
        new_name=None.

        **Example usage:**

            1. ``aman.move('hwp_angle', None)``
                Deletes the field ``hwp_angle`` from ``aman``.
            2. ``aman.move('hwp_angle', 'angle')``
                Renames the field ``hwp_angle`` to ``angle``.
            3. ``aman.move('preprocess.t2p.t2p_stats', None)``
                Deletes the field ``t2p_stats`` from the sub-AxisManager
                ``aman.preprocess.t2p``.

        """
        if name and '.' in name:
            tmp, name = name.rsplit('.', 1)
            aman = self.get(tmp)
        else:
            aman = self
        if new_name and '.' in new_name:
            tmp, new_name = new_name.rsplit('.', 1)
            new_aman = self.get(tmp)
        else:
            new_aman = self

        if new_name is None:
            del aman._fields[name]
            del aman._assignments[name]
        else:
            new_aman._fields[new_name] = aman._fields.pop(name)
            new_aman._assignments[new_name] = aman._assignments.pop(name)
        return self

    def add_axis(self, a):
        assert isinstance( a, AxisInterface)
        self._axes[a.name] = a.copy()

    def __contains__(self, name):
        attrs = name.split(".")
        tmp_item = self
        while attrs:
            attr_name = attrs.pop(0)
            if attr_name in tmp_item._fields:
                tmp_item = tmp_item._fields[attr_name]
            elif attr_name in tmp_item._axes:
                tmp_item = tmp_item._axes[attr_name]
            else:
                return False
        return True

    def __getitem__(self, name):

        # We want to support options like:
        # aman.focal_plane.xi . aman['focal_plane.xi']
        # We will safely assume that a getitem will always have '.' as the separator
        attrs = name.split(".")
        tmp_item = self
        while attrs:
            attr_name = attrs.pop(0)
            if attr_name in tmp_item._fields:
                tmp_item = tmp_item._fields[attr_name]
            elif attr_name in tmp_item._axes:
                tmp_item = tmp_item._axes[attr_name]
            else:
                raise KeyError(attr_name)
        return tmp_item

    def __setitem__(self, name, val):

        last_pos = name.rfind(".")
        val_key = name
        tmp_item = self
        if last_pos > -1:
            val_key = name[last_pos + 1:]
            attrs = name[:last_pos]
            tmp_item = self[attrs]

        if isinstance(val, AxisManager) and isinstance(tmp_item, AxisManager):
            raise ValueError("Cannot assign AxisManager to AxisManager. Please use wrap method.")

        if val_key in tmp_item._fields:
            tmp_item._fields[val_key] = val
        else:
            raise KeyError(val_key)

    def __setattr__(self, name, value):
        # Assignment to members update those members
        # We will assume that a path exists until the last member.
        # If any member prior to that does not exist a keyerror is raised.
        if "_fields" in self.__dict__ and name in self._fields.keys():
            self._fields[name] = value
        else:
            # Other assignments update this object
            self.__dict__[name] = value

    def __delattr__(self, name):
        del self._fields[name]
        del self._assignments[name]

    def __getattr__(self, name):
        # Prevent members from override special class members.
        if name.startswith("__"): raise AttributeError(name)
        try:
            val = self[name]
        except KeyError as ex:
            raise AttributeError(name) from ex
        return val

    def __dir__(self):
        return sorted(tuple(self.__dict__.keys()) + tuple(self.keys()))

    def keys(self):
        return list(self._fields.keys()) + list(self._axes.keys())

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def shape_str(self, name):
        if np.isscalar(self._fields[name]) or self._fields[name] is None:
            return ''
        s = []
        for n, ax in zip(self._fields[name].shape, self._assignments[name]):
            if ax is None:
                s.append('%i' % n)
            else:
                s.append('%s' % ax)
        return ','.join(s)

    def __repr__(self):
        def branch_marker(name):
            return '*' if isinstance(self._fields[name], AxisManager) else ''
        stuff = (['%s%s[%s]' % (k, branch_marker(k), self.shape_str(k))
                  for k in self._fields.keys()]
                 + ['%s:%s' % (k, v._minirepr_())
                    for k, v in self._axes.items()])
        return ("{}(".format(type(self).__name__)
                + ', '.join(stuff).replace('[]', '') + ")")

    @staticmethod
    def concatenate(items, axis=0, other_fields='exact'):
        """Concatenate multiple AxisManagers along the specified axis, which
        can be an integer (corresponding to the order in
        items[0]._axes) or the string name of the axis.

        This operation is difficult to sanity check so it's best to
        use it only in simple, controlled cases!  The first item is
        given significant privilege in defining what fields are
        relevant.  Fields that appear in the first item, but do not
        share the target axis, will be treated as follows depending on
        the value of other_fields:

        - If other_fields='exact' will compare entries in all items
          and if they're identical will add it. Otherwise will fail with
          a ValueError.
        - If other_fields='fail', the function will fail with a ValueError.
        - If other_fields='first', the values from the first element
          of items will be copied into the output.
        - If other_fields='drop', the fields will simply be ignored
          (and the output will only contain fields that share the
          target axis).

        """
        assert other_fields in ['exact', 'fail', 'first', 'drop']
        if not isinstance(axis, str):
            axis = list(items[0]._axes.keys())[axis]
        fields = []
        for name in items[0]._fields.keys():
            ax_dim = None
            for i, ax in enumerate(items[0]._assignments[name]):
                if ax == axis:
                    if ax_dim is not None:
                        raise ValueError('Entry %s has axis %s on more than '
                                         '1 dimension.' % (name, axis))
                    ax_dim = i
            if ax_dim is not None:
                fields.append((name, ax_dim))
        # Design the new axis.
        vals = np.hstack([item._axes[axis].vals for item in items])
        new_ax = LabelAxis(axis, vals)
        # Concatenate each entry.
        new_data = {}
        for name, ax_dim in fields:
            shape0 = None
            keepers = []
            for item in items:
                shape1 = list(item._fields[name].shape)
                if 0 in shape1:
                    continue
                shape1[ax_dim] = -1  # This dim doesn't have to match.
                if shape0 is None:
                    shape0 = shape1
                elif shape0 != shape1:
                    raise ValueError('Field %s has incompatible shapes: '
                                     % name
                                     + '%s and %s' % (shape0, shape1))
                keepers.append(item._fields[name])
            if len(keepers) == 0:
                # Well we tried.
                keepers = [items[0]._fields[name]]
            # Call class-specific concatenation if needed.
            if isinstance(keepers[0], AxisManager):
                new_data[name] = AxisManager.concatenate(
                    keepers, axis=ax_dim, other_fields=other_fields)
            elif isinstance(keepers[0], np.ndarray):
                new_data[name] = np.concatenate(keepers, axis=ax_dim)
            elif isinstance(keepers[0], csr_array):
                # Note in scipy 1.11 the default format for vstack
                # and/or hstack seems to have change, as we started
                # seeing induced cso format here.  Force preservation
                # of incoming format.
                if ax_dim == 0:
                    new_data[name] = sparse.vstack(keepers, format=keepers[0].format)
                elif ax_dim == 1:
                    new_data[name] = sparse.hstack(keepers, format=keepers[0].format)
                else:
                    raise ValueError('sparse arrays cannot concatenate along '
                                     f'axes greater than 1, received {ax_dim}')
            else:
                # The general compatible object should have a static
                # method called concatenate.
                new_data[name] = keepers[0].concatenate(keepers, axis=ax_dim)

        # Construct.
        new_axes = []
        for ax_name, ax_def in items[0]._axes.items():
            if ax_name == axis:
                ax_def = new_ax
            new_axes.append(ax_def)
        output = AxisManager(*new_axes)
        for k, v in items[0]._assignments.items():
            axis_map = [(i, n) for i, n in enumerate(v) if n is not None]
            if isinstance(items[0][k], AxisManager):
                axis_map = None  # wrap doesn't like this.
            if k in new_data:
                output.wrap(k, new_data[k], axis_map)
            else:
                if other_fields == "exact":
                    err_msg = (f"The field '{k}' does not share axis '{axis}'; " 
                              f"{k} is not identical across all items " 
                              f"pass other_fields='drop' or 'first' or else " 
                              f"remove this field from the targets.")

                    if np.any([np.isscalar(i[k]) for i in items]):
                        # At least one is a scalar...
                        if not np.all([np.isscalar(i[k]) for i in items]):
                            raise ValueError(err_msg)
                        if not np.all([_member_equal(i[k], items[0][k])
                                       for i in items[1:]]):
                            raise ValueError(err_msg)
                        output.wrap(k, items[0][k], axis_map)
                        continue

                    elif not np.all([i[k].shape==items[0][k].shape for i in items]):
                        # Has shape; shapes differ.
                        raise ValueError(err_msg)

                    elif not np.all([_member_equal(i[k], items[0][k])
                                     for i in items[1:]]):
                        # All have same shape; values not equal.
                        raise ValueError(err_msg)

                    output.wrap(k, items[0][k].copy(), axis_map)

                elif other_fields == 'fail':
                    raise ValueError(
                        f"The field '{k}' does not share axis '{axis}'; "
                        f"pass other_fields='drop' or 'first' or else "
                        f"remove this field from the targets.")
                elif other_fields == 'first':
                    # Just copy it.
                    if np.isscalar(items[0][k]):
                        output.wrap(k, items[0][k], axis_map)
                    else:
                        output.wrap(k, items[0][k].copy(), axis_map)
                elif other_fields == 'drop':
                    pass
        return output

    # Add and remove data while maintaining internal consistency.

    def wrap(self, name, data, axis_map=None,
             overwrite=False, restrict_in_place=False):
        """Add data into the AxisManager.

        Arguments:

          name (str): name of the new data.

          data: The data to register.  This must be of an acceptable
            type, i.e. a numpy array or another AxisManager.
            If scalar (or None) then data will be directly added to
            _fields with no associated axis.

          axis_map: A list that assigns dimensions of data to
            particular Axes.  Each entry in the list must be a tuple
            with the form (dim, name) or (dim, ax), where dim is the
            index of the dimension being described, name is a string
            giving the name of an axis already described in the
            present object, and ax is an AxisInterface object.
            
          overwrite (bool): If True then will write over existing data
            in field ``name`` if present.

          restrict_in_place (bool): If True, then a wrapped
            AxisManager may be modified and added, without a copy
            first.  This can be much faster, if there's no need to
            preserve the wrapped item.

        """
        if overwrite and (name in self._fields):
            self.move(name, None)
        # Don't permit AxisManager reference loops!
        if isinstance(data, AxisManager):
            assert(id(self) not in data._managed_ids())
            assert(axis_map is None)
            axis_map = [(i, v) for i, v in enumerate(data._axes.values())]
        # Handle scalars
        if np.isscalar(data) or data is None:
            if name in self._fields:
                raise ValueError(f'Key: {name} already found in {self}')
            if np.iscomplex(data):
                # Complex values aren't supported by HDF scheme right now.
                raise ValueError(f'Cannot store complex value as scalar.')
            if isinstance(data, (np.integer, np.floating, np.str_, np.bool_)):
                # Convert sneaky numpy scalars to native python int/float/str
                data = data.item()
            self._fields[name] = data
            self._assignments[name] = []
            return self
        # Promote input data to a full AxisManager, so we can call up
        # to self.merge.
        helper = AxisManager()
        assign = [None for s in data.shape]
        # Resolve each axis declaration into an axis object, and check
        # for conflict.  If an axis is passed by name only, the
        # dimensions must agree with self.  If a full axis definition
        # is included, then intersection will be performed, later.
        if axis_map is not None:
            for index, axis in axis_map:
                if not isinstance(axis, AxisInterface):
                    # So it better be a string label... that we've heard of.
                    if axis not in self._axes:
                        raise ValueError("Axis assignment refers to unknown "
                                         "axis '%s'." % axis)
                    axis = self._axes[axis]
                axis = axis.resolve(data, index)
                helper._axes[axis.name] = axis
                assign[index] = axis.name
        helper._fields[name] = data
        helper._assignments[name] = assign
        return self.merge(helper, restrict_in_place=restrict_in_place)

    def wrap_new(self, name, shape=None, cls=None, **kwargs):
        """Create a new object and wrap it, with axes mapped.  The shape can
        include axis names instead of ints, and that will cause the
        new object to be dimensioned properly and its axes mapped.

        Args:

          name (str): name of the new data.

          shape (tuple of int and std): shape in the same sense as
            numpy, except that instead of int it is allowed to pass
            the name of a managed axis.

          cls (callable): Constructor that should be used to construct
            the object; it will be called with all kwargs passed to
            this function, and with the resolved shape as described
            here.  Defaults to numpy.ndarray.

        Examples:

            Construct a 2d array and assign it the name
            'boresight_quat', with its first axis mapped to the
            AxisManager tod's "samps" axis:

            >>> tod.wrap_new('boresight_quat', shape=('samps', 4), dtype='float64')

            Create a new empty RangesMatrix, carrying a per-det, per-samp flags:

            >>> tod.wrap_new('glitch_flags', shape=('dets', 'samps'),
                             cls=so3g.proj.RangesMatrix.zeros)

        """
        if cls is None:
            cls = np.zeros
        # Turn the shape into a tuple of ints and an axis map.
        shape_ints, axis_map = [], []
        for dim, s in enumerate(shape):
            if isinstance(s, int):
                shape_ints.append(s)
            elif isinstance(s, str):
                if s in self._axes:
                    shape_ints.append(self._axes[s].count)
                    axis_map.append((dim, self._axes[s]))
                else:
                    raise ValueError(f'shape includes axis "{s}" which is '
                                     f'not in _axes: {self._axes}')
            elif isinstance(s, AxisInterface):
                # Sure, why not.
                shape_ints.append(s.count)
                axis_map.append((dim, s.copy()))
        data = cls(shape=shape_ints, **kwargs)
        return self.wrap(name, data, axis_map=axis_map)[name]

    def restrict_axes(self, axes, in_place=True):
        """Restrict this AxisManager by intersecting it with a set of Axis
        definitions.

        Arguments:
          axes (list or dict of Axis):
          in_place (bool): If in_place == True, the intersection is
            applied to self.  Otherwise, a new object is returned,
            with data copied out.

        Returns:
          The restricted AxisManager.
        """
        if in_place:
            dest = self
        else:
            dest = self.copy(axes_only=True)
            dest._assignments.update(self._assignments)
        sels = {}
        # If simple list/tuple of Axes is passed in, convert to dict
        if not isinstance(axes, dict):
            axes = {ax.name: ax for ax in axes}
        axes = {k: v for k, v in axes.items()
                if k in dest._axes and dest._axes[k] != v}
        for name, ax in axes.items():
            if name not in dest._axes:
                continue
            if dest._axes[name].count is None:
                dest._axes[name] = ax
                continue
            _, sel0, sel1 = ax.intersection(dest._axes[name], True)
            sels[name] = sel1
            dest._axes[ax.name] = ax
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                if len(axes) == 0 and in_place:
                    dest._fields[k] = v
                else:
                    dest._fields[k] = v.restrict_axes(axes, in_place=in_place)
            elif np.isscalar(v) or v is None:
                dest._fields[k] = v
            else:
                # I.e. an ndarray.
                sslice = [sels.get(ax, slice(None))
                          for ax in dest._assignments[k]]
                sslice = tuple(dest._broadcast_selector(sslice))
                sslice = simplify_slice(sslice, v.shape)
                dest._fields[k] = v[sslice]
        return dest

    def reindex_axis(self, axis, indexes, in_place=True):
        """
        Reindexes all data that is assigned to a specified axis
        with a new list/array of indexes.
        This is particularly useful if the number of detectors
        between the meta and obs data don't match.
        This function will recursively delve through all
        AxisManagers in aman and will reindex every
        data array that is found assigned to an axis
        matching the specified axis.

        Args:
            axis (str): The name of the axis in the aman to reindex.
            indexes (int array): an array of ints with length
                equal to the length of the new array
                and values equal to the idxs of the
                values in the data to be reindexed.
                Indexes that should be left as nan in
                the new array should be set to -1 or nan.

            For example:
                data = [1,3,5], indexes = [0, -1, 2, 1]
                would result in new_data = [1, nan, 5, 3]
            
            in_place (bool): If in_place == True, the intersection is
            applied to self.  Otherwise, a new object is returned,
            with data copied out.
        """
        # Check if axis even exists first
        if axis not in self._axes.keys():
            raise ValueError(f"Axis doesn't exist in aman! \
                             Can't re-index along {axis}")

        if in_place:
            aman = self
        else:
            aman = self.copy(axes_only=True)
            aman._assignments.update(self._assignments)

        # Loop through ever assignment and reindex along
        # each that is tied to the axis in question
        new_axes = {}
        reindexed_vs = {}
        assignments = list(aman._assignments.keys())
        for assignment in assignments:
            axes = aman._assignments[assignment]
            # If this assignment isn't connected to our axis
            # we can skip it.
            if axis not in axes:
                continue

            v = aman[assignment]

            if isinstance(v, AxisManager):
                # If we hit an axis manager,
                # recursively reindex it as well. Scary!
                new_v = v.reindex_axis(axis, indexes)

            else:
                # By this point we have a non AxisManager
                # assignment assigned to only our axis.
                # Build new array with the correct indexes.
                shape = [len(indexes)]
                if isinstance(v, np.ndarray):
                    for s in np.shape(v)[1:]:
                        shape.append(s)

                new_v = np.empty(shape, dtype=v.dtype)
                if isinstance(v.dtype, float):
                    # Fill any float arrays with nans
                    # Non float arrays may have weird
                    # behavior for newly added indexes. 
                    # Oh well.
                    new_v *= np.nan  

                for i, index in enumerate(indexes):
                    if np.isnan(index) or not (0 <= index < len(v)):
                        continue

                    new_v[i] = v[int(index)]

            reindexed_vs[assignment] = new_v
            new_axes[assignment] = np.array(axes)

            # Destroy the old assignment
            aman.move(name=assignment, new_name=None)

        old_axis = aman._axes[axis]

        # Recreate the axis
        if isinstance(old_axis, IndexAxis):
            # Build a new axis that has a length equal to the indexes arg.
            new_axis = IndexAxis(name=axis, count=len(indexes))

        if isinstance(old_axis, LabelAxis):
            # A LabelAxis dtype may vary by length,
            # we'll insert empty values for the newly added idxs.
            # This will produce empty strings
            # ('') for det_ids, readout_ids, etc.
            # It may produce strange behavior
            # for non string like objects. Be careful!
            vals = np.empty(len(indexes), dtype=old_axis.vals.dtype)
            for i, index in enumerate(indexes):
                if np.isnan(index) or not (0 <= int(index) < len(old_axis.vals)):
                    continue
                vals[i] = old_axis.vals[int(index)]

            new_axis = LabelAxis(name=axis, vals=vals)

        if isinstance(old_axis, OffsetAxis):
            new_axis = OffsetAxis(count=len(indexes),
                                  offset=old_axis.offset,
                                  origin_tag=old_axis.origin_tag)

        # We're done with this old axis now, destroy it.
        del aman._axes[axis]
        # Add in the reindexed axis.
        aman.add_axis(new_axis)

        # Now we'll go through all the reindexed data and wrap it back in.
        for assignment, axes in new_axes.items():
            # Build the axis map for wrapping the data.
            ax_map = []
            for i, ax in enumerate(axes):
                # Axis map looks like a list of numbered tuples.
                ax_map.append((i, ax))

            vs = reindexed_vs[assignment]
            # Need to wrap aman's with no axismap
            if isinstance(vs, AxisManager):
                aman.wrap(name=assignment, data=vs)

            else:  # Everything else needs an axismap
                aman.wrap(name=assignment, data=vs, axis_map=ax_map)

        # Everything is now reindexed and rewrapped. Done!
        return aman  # Return for rewrapping if recursively called.

    @staticmethod
    def _broadcast_selector(sslice):
        """sslice is a list of selectors, which will typically be slice(), or
        an array of indexes.  Returns a similar list of selectors, but
        with any indexing selectors promoted to a higher
        dimensionality so that the output object will be broadcast to
        the desired shape.

        For example if the input is

           (array([0,1]), slice(0,100,2), array([12,13,14]))

        then the output will be

           (array([[0],[1]]), slice(0,100,2), array([12,13,14]))

        and the result can then be used to index an array and produce
        a view with shape (2,50,3).

        """
        ex_dim = 0
        output = [s for s in sslice]
        for i in range(len(sslice) - 1, -1, -1):
            if isinstance(sslice[i], np.ndarray):
                output[i] = sslice[i].reshape(sslice[i].shape + (1,)*ex_dim)
                ex_dim += 1
        return tuple(output)

    def restrict(self, axis_name, selector, in_place=True):
        """Restrict the AxisManager by selecting a subset of items in some
        Axis.  The Axis definition and all data fields mapped to that
        axis will be modified.
        
        Arguments:
          axis_name (str): The name of the Axis.
          selector (slice or special): Selector, in a form understood
            by the underlying Axis class (see the .restriction method
            for the Axis).
          in_place (bool): If True, modifications are made to this
            object.  Otherwise, a new object with the restriction
            applied is returned.
        
        Returns:
          The AxisManager with restrictions applied.
        
        """
        if in_place:
            dest = self
        else:
            dest = self.copy(axes_only=True)
            dest._assignments.update(self._assignments)
        new_ax, sl = dest._axes[axis_name].restriction(selector)
        for k, v in self._fields.items():
            if isinstance(v, AxisManager):
                dest._fields[k] = v.copy()
                if axis_name in v._axes:
                    dest._fields[k].restrict(
                        axis_name, 
                        selector, 
                        ## copies of axes made above
                        in_place=True 
                    )
            elif np.isscalar(v) or v is None:
                dest._fields[k] = v
            else:
                sslice = [sl if n == axis_name else slice(None)
                          for n in dest._assignments[k]]
                sslice = dest._broadcast_selector(sslice)
                if in_place:
                    dest._fields[k] = v[sslice]
                else:
                    dest._fields[k] = v[sslice].copy()
        dest._axes[axis_name] = new_ax
        return dest

    @staticmethod
    def intersection_info(*items):
        """Given a list of AxisManagers, scan the axes and combine (intersect)
        any common axes.  Returns a dict that maps axis name to
        restricted Axis object.

        """
        # Get the strictest intersection of each axis.
        axes_out = OrderedDict()
        for aman in items:
            for ax in aman._axes.values():
                if ax.count is None:
                    continue
                if ax.name not in axes_out:
                    axes_out[ax.name] = ax.copy()
                elif axes_out[ax.name] != ax:
                    axes_out[ax.name] = axes_out[ax.name].intersection(
                        ax, False)
        return axes_out

    def merge(self, *amans, restrict_in_place=False):
        """Merge the data from other AxisMangers into this one.  Axes with the
        same name will be intersected.

        If restrict_in_place=True, then the amans may be modified as
        they are added to the output objcet.  When that arg is False,
        the incoming amans are all copied, even if no modifications
        are needed.

        """
        # Before messing with anything, check for key interference.
        fields = set(self._fields.keys())
        for aman in amans:
            newf = set(aman._fields.keys())
            both = fields.intersection(newf)
            if len(both):
                raise ValueError(f'Key conflict: more than one merge target '
                                 f'shares keys: {both}')
            fields.update(newf)

        # Get the intersected axis descriptions.
        axes_out = self.intersection_info(self, *amans)
        # Reduce the data in self, update our axes.
        self.restrict_axes(axes_out)
        # Import the other ones.
        for aman in amans:
            aman = aman.restrict_axes(axes_out, in_place=restrict_in_place)
            for k, v in aman._axes.items():
                if k not in self._axes:
                    self._axes[k] = v
            for k, v in aman._fields.items():
                assert(k not in self._fields)  # Should have been caught in pre-check
                self._fields[k] = v
            self._assignments.update(aman._assignments)
        return self

def simplify_slice(sslice, shape):
    """Given a tuple of slices, such as what __getitem__ might produce, and the
    shape of the array it would be applied to, return a new tuple of slices that
    accomplices the same thing, but while avoiding costly general slices if possible."""
    res = []
    for n, s in zip(shape, sslice):
        # Numpy arrays slicing is expensive, and unnecessary if they just select
        # the same elemnts in the same order
        if isinstance(s, np.ndarray):
            # Is this a trivial numpy slice? If so, replace it
            if s.size == n and np.all(s == np.arange(n)):
                res.append(slice(None))
            # Otherwise bail, and keep the whole original
            else:
                return sslice
        # For anything else just pass it through. This includes normal slices
        else: res.append(s)
    return tuple(res)

class ResultSet(object):
    """ResultSet is a special container for holding the results of
    database queries, i.e. columnar data.  The repr of a ResultSet
    states the name of its columns, and the number of rows::

      >>> print(rset)
      ResultSet<[array_code,freq_code], 17094 rows>

    You can access the column names in .keys::

      >>> print(rset.keys)
      ['array_code', 'freq_code']

    You can request a column by name, and a numpy array of values will
    be constructed for you:

      >>> rset['array_code']
      array(['LF1', 'LF1', 'LF1', ..., 'LF1', 'LF1', 'LF1'], dtype='<U3')

    You can request a row by number, and a dict will be constructed
    for you:

      >>> rset[10]
      {'base.array_code': 'LF1', 'base.freq_code': 'f027'}

    Note that the array or dict returned by indexing the ResultSet
    present copies of the data, not changing those objects will not
    update the original ResultSet.

    You can also access the raw row data in .rows, which is a simple
    list of tuples.  If you want to edit the data in a ResultSet,
    modify those data rows directly, or else use ``.asarray()`` to get
    a numpy array, modify the result, and create and a new ResultSet
    from that using the ``.from_friend`` constructor.

    You can get a structured numpy array using:

      >>> ret.asarray()
      array([('LF1', 'f027'), ('LF1', 'f027'), ('LF1', 'f027'), ...,
              ('LF1', 'f027'), ('LF1', 'f027'), ('LF1', 'f027')],
            dtype=[('array_code', '<U3'), ('freq_code', '<U4')])

    Slicing works along the row axis; and you can combine two results.
    So you could reorganize results like this, if you wanted:

      >>> rset[::2] + rset[1::2]
      ResultSet<[array_code,freq_code], 17094 rows>

    Finally, the .distinct() method returns a ResultSet containing the
    distinct elements:

      >>> rset.distinct()
      ResultSet<[array_code,freq_code], 14 rows>

    """

    #: Once instantiated, a list of the names of the ResultSet
    #: columns.
    keys = None

    #: Once instantiated, a list of the raw data tuples.
    rows = None

    def __init__(self, keys, src=None):
        self.keys = list(keys)
        if src is None:
            self.rows = []
        else:
            self.rows = [tuple(x) for x in src]

    @classmethod
    def from_friend(cls, source):
        """Return a new ResultSet populated with data from source.

        If source is a ResultSet, a copy is made.  If source is a
        numpy structured array, the ResultSet is constructed based on
        the dtype names and rows of source.

        Otherwise, a TypeError is raised.

        """
        if isinstance(source, np.ndarray):
            keys = source.dtype.names  # structured array?
            return cls(keys, list(source))
        if isinstance(source, ResultSet):
            return cls(source.keys, source.rows)
        raise TypeError(f"No implementation to construct {cls} from {source.__class__}.")

    def copy(self):
        return self.__class__(self.keys, self.rows)

    def subset(self, keys=None, rows=None):
        """Returns a copy of the object, selecting only the keys and rows
        specified.

        Arguments:
          keys: a list of keys to keep.  None keeps all.

          rows: a list or array of the integers representing which
            rows to keep.  This can also be specified as an array of
            bools, of the same length as self.rows, to select row by
            row.  None keeps all.

        """
        if keys is None:
            keys = self.keys
            def key_sel_func(row):
                return row
        else:
            key_idx = [self.keys.index(k) for k in keys]
            def key_sel_func(row):
                return [row[i] for i in key_idx]
        if rows is None:
            new_rows = map(key_sel_func, self.rows)
        elif isinstance(rows, np.ndarray) and rows.dtype == bool:
            assert(len(rows) == len(self.rows))
            new_rows = [key_sel_func(r) for r, s in zip(self.rows, rows) if s]
        else:
            new_rows = [key_sel_func(self.rows[i]) for i in rows]
        return self.__class__(keys, new_rows)

    @classmethod
    def from_cursor(cls, cursor, keys=None):
        """Create a ResultSet using the results stored in cursor, an
        sqlite.Cursor object.  The cursor must have be configured so
        that .description is populated.

        """
        if keys is None:
            keys = [c[0] for c in cursor.description]
        self = cls(keys)
        self.rows = [tuple(r) for r in cursor]
        return self

    def asarray(self, simplify_keys=False, hdf_compat=False):
        """Get a numpy structured array containing a copy of this data.  The
        names of the fields are taken from self.keys.

        Args:
          simplify_keys: If True, then the keys are stripped of any
            prefix (such as 'base.').  This is mostly for DetDb, where
            the table name can be annoying.  An error is thrown if
            this results in duplicate field names.
          hdf_compat: If True, then 'U'-type columns (Unicode strings)
            are converted to 'S'-type (byte strings), so it can be
            stored in an HDF5 dataset.

        """
        keys = [k for k in self.keys]
        if simplify_keys:  # remove prefixes
            keys = [k.split('.')[-1] for k in keys]
            assert(len(set(keys)) == len(keys))  # distinct.
        columns = tuple(map(_smart_array_cast, zip(*self.rows)))
        if hdf_compat:
            # Translate any Unicode columns to strings.
            new_cols = []
            for c in columns:
                if c.dtype.char == 'U':
                    new_cols.append(c.astype('S'))
                else:
                    new_cols.append(c)
            columns = new_cols
        dtype = [(k, c.dtype, c.shape[1:]) for k, c in zip(keys, columns)]
        output = np.ndarray(shape=len(columns[0]), dtype=dtype)
        for k, c in zip(keys, columns):
            output[k] = c
        return output

    def distinct(self):
        """
        Returns a ResultSet that is a copy of the present one, with
        duplicates removed.  The rows are sorted (according to python
        sort).
        """
        return self.__class__(self.keys, sorted(list(set(self.rows))))

    def strip(self, patterns=[]):
        """For any keys that start with a string in patterns, remove that
        string prefix from the key.  Operates in place.

        """
        for i, k in enumerate(self.keys):
            for p in patterns:
                if k.startswith(p):
                    self.keys[i] = k[len(p):]
                    break
        assert(len(self.keys) == len(set(self.keys)))

    def to_axismanager(self, axis_name="dets", axis_key="dets"):
        """Build an AxisManager directly from a ResultSet, projecting all columns
        along a single axis. This requires no additional metadata to build
        
        Args:
            axis_name: string, name of the axis in the AxisManager
            axis_key: string, name of the key in the ResultSet to put into the
                axis labels. This key will not be added to the AxisManager
                fields. 
        """
        from sotodlib import core
        aman = core.AxisManager(
            core.LabelAxis(axis_name, self[axis_key])
        )
        for k in self.keys:
            if k == axis_key:
                continue
            if any([ x is None for x in self[k]]):
                raise TypeError("None(s) found in key {}, these cannot be ".format(k)+
                               "nicely wrapped into an AxisManager")
            aman.wrap(k, self[k], [(0,axis_name)])
        return aman

    def restrict_dets(self, restriction, detdb=None):
        # There are 4 classes of keys:
        # - dets:* keys appearing only in restriction
        # - dets:* keys appearing only in self
        # - dets:* keys appearing in both
        # - other.
        new_keys = [k for k in restriction if k.startswith('dets:')]
        match_keys = []
        for k in self.keys:
            if k in new_keys:
                match_keys.append(k)
                new_keys.remove(k)
        other_keys = [k for k in self.keys if k not in match_keys]
        output_keys = new_keys + match_keys + other_keys # disjoint.
        output_rows = []
        for row in self:
            row = dict(row)  # copy
            for k in match_keys:
                if row[k] != restriction[k]:
                    break
            else:
                # You passed.
                row.update({k: restriction[k] for k in new_keys})
            output_rows.append([row[k] for k in output_keys])
        # That's all.
        return self.__class__(output_keys, output_rows)

    # Everything else is just implementing container-like behavior

    def __repr__(self):
        keystr = 'empty'
        if self.keys is not None:
            keystr = ','.join(self.keys)
        return ('{}<[{}], {} rows>'.format(self.__class__.__name__,
                                           keystr, len(self)))

    def __len__(self):
        return len(self.rows)

    def append(self, item):
        vals = []
        for k in self.keys:
            if k not in item.keys():
                raise ValueError(f"Item to append must include key '{k}'")
            vals.append(item[k])
        self.rows.append(tuple(vals))

    def extend(self, items):
        if not isinstance(items, ResultSet):
            raise TypeError("Extension only valid for two ResultSet objects.")
        if self.keys != items.keys:
            raise ValueError("Keys do not match: {} <- {}".format(
                self.keys, items.keys))
        self.rows.extend(items.rows)

    def __getitem__(self, item):
        # Simple row look-up... convert to dict.
        if isinstance(item, int) or isinstance(item, np.integer):
            return OrderedDict([(k,v) for k, v in
                                zip(self.keys, self.rows[item])])
        # Look-up by column...
        if isinstance(item, str):
            index = self.keys.index(item)
            return _smart_array_cast([x[index] for x in self.rows],
                                     field_detail=f"Key {item}:")
        # Slicing.
        output = self.__class__(self.keys, self.rows[item])
        return output

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __add__(self, other):
        output = self.copy()
        output += other
        return output

    @staticmethod
    def concatenate(items, axis=0):
        assert(axis == 0)
        output = items[0].copy()
        for item in items[1:]:
            output += item
        return output

    def merge(self, src):
        """Merge with src, which must have same number of rows as self.
        Duplicate columns are not allowed.

        """
        if len(self) != len(src):
            raise ValueError("self and src have different numbers of rows.")
        for k in src.keys:
            if k in self.keys:
                raise ValueError("Duplicate key: %s" % k)
        new_keys = self.keys + src.keys
        new_rows = [r0 + r1 for r0, r1 in zip(self.rows, src.rows)]
        self.keys, self.rows = new_keys, new_rows


def _smart_array_cast(values, dtype=None, field_detail=None):
    """Convert a list of values to a numpy array.  Let numpy casting do
    its job, but replace any Nones in the list with some better value,
    first.  The better value is nan, for floats.  For strings and
    ints, the best we can do is '' and 0 -- in which case a warning is
    issued.

    For string and int table columns it would be much better to set
    sensible default values in the schema rather than have null vals
    persist in the db.

    """
    non_null = [v for v in values if v is not None]
    if len(non_null) == 0:
        return np.full(len(values), np.nan)
    if len(non_null) == len(values):
        return np.array(values, dtype=dtype)

    warn = None
    trial = np.array(non_null)
    if trial.dtype == np.float32:
        fill = np.float32(np.nan)
    elif trial.dtype == np.float64:
        fill = np.float64(np.nan)
    elif np.issubdtype(trial.dtype, np.str_):
        fill = ''
        warn = 'Replacing null entries with "".'
    elif np.issubdtype(trial.dtype, np.integer):
        fill = 0
        warn = 'Replacing null entries with 0.'
    else:
        fill = None
        warn = 'No patch value for null (dtype=%s).' % (str(trial.dtype))

    revalues = np.array([fill if v is None else v
                         for v in values])
    if revalues.dtype != trial.dtype:
        warn = ('' if warn is None else warn + ' ') + \
        warnings.warn("Unexpected dtype change.")

    if warn:
        if field_detail:
            warn = field_detail + ' ' + warn
        warnings.warn(warn)

    return revalues

class ObsFileDb:
    """Stub with the small subset of ObsFileDb functionality we use in sogma"""
    def __init__(self, map_file):
        if isinstance(map_file, sqlite3.Connection):
            self.conn = map_file
        else:
            self.conn = sqlite3.connect(map_file)
        self.conn.row_factory = sqlite3.Row  # access columns by name
        self.prefix = os.path.split(os.path.abspath(map_file))[0] + '/'
    # Retrieval
    def get_detsets(self, obs_id):
        """Returns a list of all detsets represented in the observation
        specified by obs_id."""
        c = self.conn.execute('select distinct detset from files where obs_id=?', (obs_id,))
        return [r[0] for r in c]
    def get_dets(self, detset):
        """Returns a list of all detectors in the specified detset."""
        c = self.conn.execute('select det from detsets where name=?', (detset,))
        return [r[0] for r in c]
    def get_files(self, obs_id, detsets=None, prefix=None):
        """Get the file names associated with a particular obs_id and detsets.
        Returns:
          OrderedDict where the key is the detset name and the value
          is a list of tuples of the form (full_filename,
          sample_start, sample_stop)."""
        if prefix is None: prefix = self.prefix
        if detsets is None:
            c = self.conn.execute('select detset, name, sample_start, sample_stop from files where obs_id=? order by detset, sample_start', (obs_id,))
        else:
            c = self.conn.execute('select detset, name, sample_start, sample_stop from files where obs_id=? and detset in (%s) order by detset, sample_start' % ','.join(['?' for _ in detsets]), (obs_id,) + tuple(detsets))
        output = OrderedDict()
        for r in c:
            if not r[0] in output:
                output[r[0]] = []
            output[r[0]].append((os.path.join(prefix, r[1]), r[2], r[3]))
        return output
