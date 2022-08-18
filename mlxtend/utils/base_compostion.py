"""Utilties to handle estimator list"""

from sklearn.utils.metaestimators import _BaseComposition


class _BaseXComposition(_BaseComposition):
    """
    parameter handler for list of estimators
    """

    def _set_params(self, attr, named_attr, **params):
        # Ordered parameter replacement
        # 1. root parameter
        if attr in params:
            setattr(self, attr, params.pop(attr))

        # 2. single estimator replacement
        items = getattr(self, named_attr)
        names = []
        if items:
            names, estimators = zip(*items)
            estimators = list(estimators)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                # replace single estimator and re-build the
                # root estimators list
                for i, est_name in enumerate(names):
                    if est_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del estimators[i]
                        else:
                            estimators[i] = new_val
                        break
                # replace the root estimators
                setattr(self, attr, estimators)

        # 3. estimator parameters and other initialisation arguments
        super(_BaseXComposition, self).set_params(**params)
        return self
