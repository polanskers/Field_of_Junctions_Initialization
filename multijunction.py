import torch
import torch.nn.functional as F

class MultiJunction:
    def __init__(self, order, dev, vertex=None, angle=None, wedges=None):
        """
        Inputs
        ------
        order     Junction order, i.e. number of wedges per junction (positive integer)
        vertex    (x,y) coordinates of vertices for each junction. Accepts an N x 2 array of coordinates
                  If this argument is omitted, the default value is (0.0,0.0).
        angle     Angles in radians representing overall orientation for each junction. Length N array.
                  If this argument is omitted, the default value is 0.0.
        wedges    Relative solid angles of wedges. 2D array of size N x order.
                  If this argument is omitted, the default value is an array of ones
                  (so all wedges have equal solid angles).
        
        Note:     The entries of "wedges" need not sum to 2*pi; they are
                  normalized internally. Also, "wedges" cannot be all zeros.
        """

        # Make sure order is a positive integer
        if isinstance(order, int):
              if order <= 0:
                raise ValueError("Junction order must be a positive integer")
        elif isinstance(order, float):
              if not order.is_integer() or order <= 0:
                raise ValueError("Junction order must be a positive integer")
        else:
              raise ValueError("Junction order must be a positive integer")
                
        self.order = int(order)
        
        # Define wedge vertex
        self.vertex = vertex if vertex is not None else torch.zeros((1,2),device=dev)

        # Define global orientation
        self.angle = angle if angle is not None else 0.0

        # Define wedge solid angles, and normalize for summing to 2*pi
        self.wedges = wedges if wedges is not None else torch.ones((1,order),device=dev)
        self.wedges = self.wedges*(2*torch.pi)/torch.sum(self.wedges,1).unsqueeze(1)
        
        padding = [(1)] + [(0)]
        
        # Compute wedge central angles
        self.centralangles = self.angle.unsqueeze(1) + self.wedges/2 + F.pad(torch.cumsum(self.wedges,dim=1)[:,:-1],padding)

        # Compute wedge boundary angles
        self.boundaryangles = self.angle.unsqueeze(1) + F.pad(torch.cumsum(self.wedges,dim=1)[:,:-1],padding)


    def render_wedges(self,opts,dev):
        """
        Test whether point (xt,yt) is in jth wedge

        Inputs
        ------
        xt, yt          xt is a 1D array of x points being tested, yt is a 1D array of y points being tested

        Output
        ------
        indicator       Binary array: N x |xt| x |yt| x J, where J is order of the junction

        """
        
        xt = torch.linspace(opts.patchmin, opts.patchmax, opts.patchres, device=dev).unsqueeze(0)
        yt = torch.linspace(opts.patchmin, opts.patchmax, opts.patchres, device=dev).unsqueeze(-1)
                
        indicator = (xt.unsqueeze(-1).unsqueeze(0) - self.vertex[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) \
                        * torch.cos(self.centralangles).unsqueeze(1).unsqueeze(1) + \
                    (yt.unsqueeze(-1).unsqueeze(0) - self.vertex[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) \
                        * torch.sin(self.centralangles).unsqueeze(1).unsqueeze(1)  > \
                    torch.cos(self.wedges/2).unsqueeze(1).unsqueeze(1) * \
                       torch.sqrt((xt.unsqueeze(-1).unsqueeze(0) - self.vertex[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**2 + \
                               (yt.unsqueeze(-1).unsqueeze(0) - self.vertex[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**2)
        
        return indicator

    def render_boundaries(self,opts,dev):
        """
        Render an image of the wedge boundaries over a square patch

        Inputs
        ------
        opts   Object with the following attributes:
               patchmin          Lower bound (xmin = ymin) of square domain
               patchmax          Upper bound (xmax = ymax) of square domain
               patchres          Number of pixels in each dimension
               delta             Dirac delta relaxation parameter for rendering boundary maps   
             
        Output
        ------
        im    Image of size [patchres, patchres] with values in [0,1]
        """

        # coordinate grid of pixel locations 
        x = torch.linspace(opts.patchmin, opts.patchmax, opts.patchres, device=dev).unsqueeze(0)
        y = torch.linspace(opts.patchmin, opts.patchmax, opts.patchres, device=dev).unsqueeze(-1)
  
        # Loop over wedge-boundaries and store their boundary maps.
        # Use [1 / (1 + (x/opts.delta)**2 )] for the relaxed dirac distribution 
        ims = 1.0 / (1.0 + (( (x.unsqueeze(-1).unsqueeze(0) - self.vertex[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * torch.cos(self.boundaryangles).unsqueeze(1).unsqueeze(1) + \
                       (y.unsqueeze(-1).unsqueeze(0) - self.vertex[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * torch.sin(self.boundaryangles).unsqueeze(1).unsqueeze(1) - \
                        torch.sqrt((x.unsqueeze(-1).unsqueeze(0) - self.vertex[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**2 + \
                                   (y.unsqueeze(-1).unsqueeze(0) - self.vertex[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**2)) \
                      / opts.delta )**2 )  
        
        # return max over boundary maps 
        return torch.amax(ims,dim=-1)