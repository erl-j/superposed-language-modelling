import torch
import einops

class HierarchicalModel(torch.nn.Module):

    def __init__(self):
        self.big_model = BigModel()
        self.small_model = SmallModel()

    def forward(self, x):
        event_z = self.big_model.event_model_forward(x)
        attr_z = self.small_model.attribute_model_forward(x, event_z)
        
        return attr_z
    

    
class BigModel(torch.nn.Module):
    def __init__(self):

        # Initialize the parent class
        super(HierarchicalModel, self).__init__()

        # embedding layer
        self.embedding = ...
        self.unembedding = ...

        self.event_pos_encoding = ...
        self.attr_pos_encoding = ...
        # causal model
        self.event_main_model = ...

        self.attribute_main_model = ...

    def event_model_forward(self, x):
        # x has shape (batch_size, num_events, num_attributes)
        attr_embed = self.embedding(x)
        # sum over the attribute dimension with einops
        attr_embed = einops.reduce(attr_embed, 'b e a -> b e', 'sum')
        # add positional encoding
        attr_embed = attr_embed + self.event_pos_encoding
        # pass through the event model
        event_z = self.event_model(attr_embed)
        return event_z
    
class SmallModel(torch.nn.Module):

    def attribute_model_forward(self, x, event_z):
        # embed
        attr_embed = self.embedding(x)
        # attribute model, not needed?
        attr_embed = attr_embed + self.attr_pos_encoding
        # pass through the main model
        attr_z = self.attribute_model(attr_embed, event_z)
        return attr_z









        