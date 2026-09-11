# Exception hierarchy for sogma
class ExpectedError(Exception): pass
class LoadError(ExpectedError): pass
class DataMissing(LoadError): pass
class NothingLeft(LoadError): pass
