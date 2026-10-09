# Exception hierarchy for sogma
class ExpectedError(Exception): pass
class  RecoverableError(ExpectedError): pass
class   LoadError(RecoverableError): pass
class    DataMissing(LoadError): pass
class    NothingLeft(LoadError): pass
class SignalOrderError(Exception): pass

def format_multi_exception(exlist):
	"""Given an exception list of the form [(label,exception),(label,exception),...],
	return a deduplicated string showing which exception messages apply to which labels"""
	from pixell import utils
	labels, exceptions = zip(*exlist)
	msgs   = [str(ex) for ex in exceptions]
	uvals, order, inds = utils.find_equal_groups_fast(msgs)
	omsgs  = []
	for ui, uval in enumerate(uvals):
		mysubs = [labels[o] for o in order[inds[ui]:inds[ui+1]]]
		omsgs.append(",".join(mysubs) + ": " + uval)
	return ", ".join(omsgs)
