from graphviz import Digraph

# Initialize the flow diagram
flow = Digraph()
flow.attr(size='10')

# Nodes
flow.node('A', 'Start Application')
flow.node('B', 'Enter Current Number of Registrations')
flow.node('C', 'Enter Days Left in Registration Period')
flow.node('D', 'Click "Calculate"')
flow.node('E', 'Display Estimated Registrations\nand Upper/Lower Bounds')
flow.node('F', 'Error: Invalid Input\nPrompt to Re-enter')
flow.node('G', 'Click "Clear" to Reset Fields')
flow.node('H', 'Interpret and Use Results')

# Edges
flow.edge('A', 'B', 'Launch GUI')
flow.edge('B', 'C', 'Input Data')
flow.edge('C', 'D', 'Complete Data Entry')
flow.edge('D', 'E', 'Valid Data')
flow.edge('D', 'F', 'Invalid Data')
flow.edge('F', 'G', 'User Action')
flow.edge('G', 'B', 'Restart Data Entry')
flow.edge('E', 'H', 'Review Output')

# Render the flowchart
flowchart_path = 'User_Interaction_Flowchart.svg'
flow.render(flowchart_path, format='svg', cleanup=True)

flowchart_path+'.svg'
