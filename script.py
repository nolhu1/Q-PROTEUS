from graphviz import Digraph

dot = Digraph(comment='Aethra Solution Outcome', format='png')
dot.attr(rankdir='TB', size='8')

dot.node('Hospitals', 'Hospitals & Research Centers', shape='box', style='filled', color='#D6EAF8')
dot.node('Platform', 'Aethra Platform\n(Federated AI Hub)', shape='ellipse', style='filled', color='#AED6F1')
dot.node('Benefits', 'Better AI Models\nImproved Diagnostics\nFaster Research', shape='box', style='filled', color='#85C1E9')

dot.edge('Hospitals', 'Platform', label='Collaborate Securely', color='blue')
dot.edge('Platform', 'Benefits', label='Shared Insights', color='darkblue')

dot.render('solution_intro_outcome', view=True)
