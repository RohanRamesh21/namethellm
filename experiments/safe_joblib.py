"""Load a joblib file from an untrusted source with an allow-list of classes (pickle can run code on load).
Only scikit-learn text vectorizers, numpy arrays/dtypes, scipy sparse matrices and a few builtins are allowed; anything else raises.
Every class the file asks for is recorded in AUDIT so the allow-list can be reviewed."""
from joblib import numpy_pickle as npk

ALLOW = {
    ('sklearn.feature_extraction.text', 'TfidfVectorizer'), ('sklearn.feature_extraction.text', 'TfidfTransformer'),
    ('sklearn.feature_extraction.text', 'CountVectorizer'),
    ('joblib.numpy_pickle', 'NumpyArrayWrapper'), ('numpy', 'dtype'), ('numpy', 'ndarray'), ('numpy', 'float32'), ('numpy', 'float64'),
    ('numpy', 'int32'), ('numpy', 'int64'), ('numpy', 'bool_'),
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('collections', 'OrderedDict'), ('collections', 'defaultdict'),
    ('scipy.sparse._csr', 'csr_matrix'), ('scipy.sparse._csc', 'csc_matrix'), ('scipy.sparse._dia', 'dia_matrix'),
    ('scipy.sparse.csr', 'csr_matrix'),
    ('builtins', 'set'), ('builtins', 'frozenset'), ('builtins', 'slice'), ('builtins', 'complex'), ('builtins', 'object'),
}
AUDIT = []


class _Safe(npk.NumpyUnpickler):
    def find_class(self, module, name):
        AUDIT.append((module, name))
        if (module, name) not in ALLOW:
            raise ValueError(f'blocked class {module}.{name} in untrusted pickle')
        return super().find_class(module, name)


def safe_load(path):
    with open(path, 'rb') as f:
        with npk._validate_fileobject_and_memmap(f, str(path), None) as (fobj, mmap_mode):
            if isinstance(fobj, str): raise ValueError('old joblib format not supported')
            return _Safe(str(path), fobj, True, mmap_mode=mmap_mode).load()
