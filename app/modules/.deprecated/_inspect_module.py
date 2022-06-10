import inspect
from mesa.space import ContinuousSpace

lines = inspect.getsource(ContinuousSpace)
print(lines)