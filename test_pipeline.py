from sentinel.pipeline import SentinelPipeline
from sentinel.models import Sender

p = SentinelPipeline()
result = p.process('I feel happy but I am nervous about the test.', Sender.USER)
print(type(result).__name__)
print('Has clauses:', len(result.clauses))
print('Clause type:', type(result.clauses[0]).__name__ if result.clauses else 'None')
print('Routes:', [(c.matrix.importance, c.route_action.value) for c in result.clauses])
print('Document score:', (result.document_score.importance, result.document_score.valence, result.document_score.arousal))
print('Document route:', result.document_route_action.value)
