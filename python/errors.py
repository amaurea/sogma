# Exception hierarchy for sogma
class ExpectedError(Exception): pass
class  RecoverableError(ExpectedError): pass
class   LoadError(RecoverableError): pass
class    DataMissing(LoadError): pass
class    NothingLeft(LoadError): pass
