from sentinel.pipeline import SentinelPipeline
from sentinel.models import Sender

p = SentinelPipeline()
result = p.process('I feel happy but I am nervous about the test.', Sender.USER)
print(type(result).__name__)
print('Has clauses:', len(result.clauses))
print('Clause type:', type(result.clauses[0]).__name__ if result.clauses else 'None')
