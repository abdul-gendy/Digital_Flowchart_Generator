
from flowgiston import FlowgistonChart, flowgiston_base

Base = flowgiston_base()
class rectangle(Base):
    fillcolor = 'white'
    fontcolor = 'black'
    shape = 'rectangle'

class diamond(Base):
    fillcolor = 'white'
    fontcolor = 'black'
    shape = 'diamond'

class triangle(Base):
    fillcolor = 'white'
    fontcolor = 'black'
    shape = 'triangle'

class circle(Base):
    fillcolor = 'white'
    fontcolor = 'black'
    shape = 'circle'

class flowchart_graphics_simple:
    """
    This is a class that takes in the classified shapes and their arrow relationships
    and does the required function calls in the Flowgiston library to construct the 
    desired flowchart

    Attributes:
        shapes_info (dict): dictionary containing all the shapes and their information with
                            added information regarding which shapes are linked to each other
    """
    def __init__(self, shapes_info):
        """
        The constructor for flowchart_graphics_simple class.
  
        Parameters:
            shapes_info (dict): dictionary containing all the shapes and their information with
                                added information regarding which shapes are linked to each other
        """
        self.shapes_info = shapes_info
        self.classes = ["circle", "rectangle", "diamond", "triangle"]
        self.chart = FlowgistonChart(Base)

    def setup_node(self, shape):
        """
        Function that helps create nodes in the flowchart
  
        Parameters:
            shape (str): required flow chart shape

        Returns:
            created_node: flogiston chart node
        """
        if shape == "circle":
            created_node = self.chart.circle.node("")
        elif shape == "rectangle":
            created_node = self.chart.rectangle.node("")
        elif shape == "diamond":
            created_node = self.chart.diamond.node("")
        elif shape == "triangle":
            created_node = self.chart.triangle.node("")
        else:
            raise ValueError("unknown shape to draw")
        return created_node

    def create_nodes_for_shapes(self):
        """
        creates node for all shapes listed in shapes_info

        Returns:
            nodes: created flowgiston chart nodes
        """
        #Go through code dict created from reconstruct_flowchart.py and create the necessary graphics
        nodes = []
        for shape_info in self.shapes_info:
            shape = shape_info["shape"]
            created_node = self.setup_node(shape)
            nodes.append(created_node)
        return nodes

    def draw_flowchart(self):
        """
        Connect created nodes based on arrow information
        and display flowchart
        """
        nodes = self.create_nodes_for_shapes()
        for i in range(0, len(self.shapes_info)):
            current_node = nodes[i]
            shapes_to_connect_to = self.shapes_info[i]["arrow_to"]
            for shape_index in shapes_to_connect_to:
                target_node = nodes[shape_index]
                self.chart.edge(current_node, target_node, '')
        self.chart.render( view=True)