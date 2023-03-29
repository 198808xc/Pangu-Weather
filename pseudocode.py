'''
Pseudocode of Pangu-Weather
'''
# The pseudocode can be implemented using deep learning libraries, e.g., Pytorch and Tensorflow or other high-level APIs

# Basic operations used in our model, namely Linear, Conv3d, Conv2d, ConvTranspose3d and ConvTranspose2d
# Linear: Linear transformation, available in all deep learning libraries
# Conv3d and Con2d: Convolution with 2 or 3 dimensions, available in all deep learning libraries
# ConvTranspose3d, ConvTranspose2d: transposed convolution with 2 or 3 dimensions, see Pytorch API or Tensorflow API
from Your_AI_Library import Linear, Conv3d, Conv2d, ConvTranspose3d, ConvTranspose2d

# Functions in the networks, namely GeLU, DropOut, DropPath, LayerNorm, and SoftMax
# GeLU: the GeLU activation function, see Pytorch API or Tensorflow API
# DropOut: the dropout function, available in all deep learning libraries
# DropPath: the DropPath function, see the implementation of vision-transformer, see timm pakage of Pytorch
# A possible implementation of DropPath: from timm.models.layers import DropPath
# LayerNorm: the layer normalization function, see Pytorch API or Tensorflow API
# Softmax: softmax function, see Pytorch API or Tensorflow API
from Your_AI_Library import GeLU, DropOut, DropPath, LayerNorm, SoftMax

# Common functions for roll, pad, and crop, depends on the data structure of your software environment
from Your_AI_Library import roll3D, pad3D, pad2D, Crop3D, Crop2D

# Common functions for reshaping and changing the order of dimensions
# reshape: change the shape of the data with the order unchanged, see Pytorch API or Tensorflow API
# TransposeDimensions: change the order of the dimensions, see Pytorch API or Tensorflow API
from Your_AI_Library import reshape, TransposeDimensions

# Common functions for creating new tensors
# ConstructTensor: create a new tensor with an arbitrary shape
# TruncatedNormalInit: Initialize the tensor with Truncate Normalization distribution
# RangeTensor: create a new tensor like range(a, b)
from Your_AI_Library import ConstructTensor, TruncatedNormalInit, RangeTensor

# Common operations for the data, you may design it or simply use deep learning APIs default operations
# LinearSpace: a tensor version of numpy.linspace
# MeshGrid: a tensor version of numpy.meshgrid
# Stack: a tensor version of numpy.stack
# Flatten: a tensor version of numpy.ndarray.flatten
# TensorSum: a tensor version of numpy.sum
# TensorAbs: a tensor version of numpy.abs
# Concatenate: a tensor version of numpy.concatenate
from Your_AI_Library import LinearSpace, MeshGrid, Stack, Flatten, TensorSum, TensorAbs, Concatenate

# Common functions for training models
# LoadModel and SaveModel: Load and save the model, some APIs may require further adaptation to hardwares
# Backward: Gradient backward to calculate the gratitude of each parameters
# UpdateModelParametersWithAdam: Use Adam to update parameters, e.g., torch.optim.Adam
from Your_AI_Library import LoadModel, Backward, UpdateModelParametersWithAdam, SaveModel

# Custom functions to read your data from the disc
# LoadData: Load the ERA5 data
# LoadConstantMask: Load constant masks, e.g., soil type
# LoadStatic: Load mean and std of the ERA5 training data, every fields such as T850 is treated as an image and calculate the mean and std
from Your_Data_Code import LoadData, LoadConstantMask, LoadStatic


def Inference(input, input_surface, forecast_range):
  '''Inference code, describing the algorithm of inference using models with different lead times. 
  PanguModel24, PanguModel6, PanguModel3 and PanguModel1 share the same training algorithm but differ in lead times.
  Args:
    input: input tensor, need to be normalized to N(0, 1) in practice
    input_surface: target tensor, need to be normalized to N(0, 1) in practice
    forecast_range: iteration numbers when roll out the forecast model
  '''

  # Load 4 pre-trained models with different lead times
  PanguModel24 = LoadModel(ModelPath24)
  PanguModel6 = LoadModel(ModelPath6)
  PanguModel3 = LoadModel(ModelPath3)
  PanguModel1 = LoadModel(ModelPath1)

  # Load mean and std of the weather data
  weather_mean, weather_std, weather_surface_mean, weather_surface_std = LoadStatic()

  # Store initial input for different models
  input_24, input_surface_24 = input, input_surface
  input_6, input_surface_6 = input, input_surface
  input_3, input_surface_3 = input, input_surface

  # Using a list to store output
  output_list = []

  # Note: the following code is implemented for fast inference of [1,forecast_range]-hour forecasts -- if only one lead time is requested, the inference can be much faster.
  for i in range(forecast_range):
    # switch to the 24-hour model if the forecast time is 24 hours, 48 hours, ..., 24*N hours
    if (i+1) % 24 == 0:
      # Switch the input back to the stored input
      input, input_surface = input_24, input_surface_24

      # Call the model pretrained for 24 hours forecast
      output, output_surface = PanguModel24(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

      # Stored the output for next round forecast
      input_24, input_surface_24 = output, output_surface
      input_6, input_surface_6 = output, output_surface
      input_3, input_surface_3 = output, output_surface

    # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
    elif (i+1) % 6 == 0:
      # Switch the input back to the stored input
      input, input_surface = input_6, input_surface_6

      # Call the model pretrained for 6 hours forecast
      output, output_surface = PanguModel6(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean
      
      # Stored the output for next round forecast
      input_6, input_surface_6 = output, output_surface
      input_3, input_surface_3 = output, output_surface

    # switch to the 3-hour model if the forecast time is 3 hours, 9 hours, ..., 6*N + 3 hours
    elif (i+1) % 3 ==0:
      # Switch the input back to the stored input
      input, input_surface = input_3, input_surface_3

      # Call the model pretrained for 3 hours forecast
      output, output_surface = PanguModel3(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean
      
      # Stored the output for next round forecast
      input_3, input_surface_3 = output, output_surface

    # switch to the 1-hour model
    else:
      # Call the model pretrained for 1 hours forecast
      output, output_surface = PanguModel1(input, input_surface)

      # Restore from uniformed output
      output = output * weather_std + weather_mean
      output_surface = output_surface * weather_surface_std + weather_surface_mean

    # Stored the output for next round forecast
    input, input_surface = output, output_surface

    # Save the output
    output_list.append((output, output_surface))
  return output_list


def Train():
  '''Training code'''
  # Initialize the model, for some APIs some adaptation is needed to fit hardwares
  model = PanguModel()

  # Train single Pangu-Weather model
  epochs = 100
  for i in range(epochs):
    # For each epoch, we iterate from 1979 to 2017
    # dataset_length is the length of your training data, e.g., the sample between 1979 and 2017
    for step in range(dataset_length):
      # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
      # Note the data need to be randomly shuffled
      # Note the input and target need to be normalized, see Inference() for details
      input, input_surface, target, target_surface = LoadData(step)

      # Call the model and get the output
      output, output_surface = model(input, input_surface)

      # We use the MAE loss to train the model
      # The weight of surface loss is 0.25
      # Different weight can be applied for differen fields if needed
      loss = TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25

      # Call the backward algorithm and calculate the gratitude of parameters
      Backward(loss)

      # Update model parameters with Adam optimizer
      # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
      # A example solution is using torch.optim.adam
      UpdateModelParametersWithAdam()

  # Save the model at the end of the training stage
  SaveModel()

class PanguModel:
  def __init__(self):
    # Drop path rate is linearly increased as the depth increases
    drop_path_list = LinearSpace(0, 0.2, 8)

    # Patch embedding
    self._input_layer = PatchEmbedding((2, 4, 4), 192)

    # Four basic layers
    self.layer1 = EarthSpecificLayer(2, 192, drop_list[:2], 6)
    self.layer2 = EarthSpecificLayer(6, 384, drop_list[6:], 12)
    self.layer3 = EarthSpecificLayer(6, 384, drop_list[6:], 12)
    self.layer4 = EarthSpecificLayer(2, 192, drop_list[:2], 6)

    # Upsample and downsample
    self.upsample = UpSample(384, 192)
    self.downsample = DownSample(192)

    # Patch Recovery
    self._output_layer = PatchRecovery(384)
    
  def forward(self, input, input_surface):
    '''Backbone architecture'''
    # Embed the input fields into patches
    x = self._input_layer(input, input_surface)

    # Encoder, composed of two layers
    # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 360, 181) 

    # Store the tensor for skip-connection
    skip = x

    # Downsample from (8, 360, 181) to (8, 180, 91)
    x = self.downsample(x, 8, 360, 181)

    # Layer 2, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 180, 91) 

    # Decoder, composed of two layers
    # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer3(x, 8, 180, 91) 

    # Upsample from (8, 180, 91) to (8, 360, 181)
    x = self.upsample(x)

    # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
    x = self.layer4(x, 8, 360, 181) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = Concatenate(skip, x)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x)
    return output, output_surface

class PatchEmbedding:
  def __init__(self, patch_size, dim):
    '''Patch embedding operation'''
    # Here we use convolution to partition data into cubes
    self.conv = Conv3d(input_dims=5, output_dims=dim, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = Conv2d(input_dims=7, output_dims=dim, kernel_size=patch_size[1:], stride=patch_size[1:])

    # Load constant masks from the disc
    self.land_mask, self.soil_type, self.topography = LoadConstantMask()
    
  def forward(self, input, input_surface):
    # Zero-pad the input
    input = Pad3D(input)
    input_surface = Pad2D(input_surface)

    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    input = self.conv(input)

    # Add three constant fields to the surface fields
    input_surface =  Concatenate(input_surface, self.land_mask, self.soil_type, self.topography)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)

    # Concatenate the input in the pressure level, i.e., in Z dimension
    x = Concatenate(input, input_surface)

    # Reshape x for calculation of linear projections
    x = TransposeDimensions(x, (0, 2, 3, 4, 1))
    x = reshape(x, target_shape=(x.shape[0], 8*360*181, x.shape[-1]))
    return x
    
class PatchRecovery:
  def __init__(self, dim):
    '''Patch recovery operation'''
    # Hear we use two transposed convolutions to recover data
    self.conv = ConvTranspose3d(input_dims=dim, output_dims=5, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = ConvTranspose2d(input_dims=dim, output_dims=4, kernel_size=patch_size[1:], stride=patch_size[1:])
    
  def forward(self, x, Z, H, W):
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions
    x = TransposeDimensions(x, (0, 2, 1))
    x = reshape(x, target_shape=(x.shape[0], x.shape[1], Z, H, W))

    # Call the transposed convolution
    output = self.conv(x[:, :, 1:, :, :])
    output_surface = self.conv_surface(x[:, :, 0, :, :])

    # Crop the output to remove zero-paddings
    output = Crop3D(output)
    output_surface = Crop2D(output_surface)
    return output, output_surface

class DownSample:
  def __init__(self, dim):
    '''Down-sampling operation'''
    # A linear function and a layer normalization
    self.linear = Linear(4*dim, 2*dim, bias=Fasle)
    self.norm = LayerNorm(4*dim)
  
  def forward(self, x, Z, H, W):
    # Reshape x to three dimensions for downsampling
    x = reshape(x, target_shape=(x.shape[0], Z, H, W, x.shape[-1]))

    # Padding the input to facilitate downsampling
    x = Pad3D(x)

    # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
    Z, H, W = x.shape
    # Reshape x to facilitate downsampling
    x = reshape(x, target_shape=(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1]))
    # Change the order of x
    x = TransposeDimensions(x, (0,1,2,4,3,5,6))
    # Reshape to get a tensor of resolution (8, 180, 91)
    x = reshape(x, target_shape=(x.shape[0], Z*(H//2)*(W//2), 4 * x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x)
    return x

class UpSample:
  def __init__(self, input_dim, output_dim):
    '''Up-sampling operation'''
    # Linear layers without bias to increase channels of the data
    self.linear1 = Linear(input_dim, output_dim*4, bias=False)

    # Linear layers without bias to mix the data up
    self.linear2 = Linear(output_dim, output_dim, bias=False)

    # Normalization
    self.norm = LayerNorm(output_dim)
  
  def forward(self, x):
    # Call the linear functions to increase channels of the data
    x = self.linear1(x)

    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
    # Reshape x to facilitate upsampling.
    x = reshape(x, target_shape=(x.shape[0], 8, 180, 91, 2, 2, x.shape[-1]//4))
    # Change the order of x
    x = TransposeDimensions(x, (0,1,2,4,3,5,6))
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = reshape(x, target_shape=(x.shape[0], 8, 360, 182, x.shape[-1]))

    # Crop the output to the input shape of the network
    x = Crop3D(x)

    # Reshape x back
    x = reshape(x, target_shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1]))

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x
  
class EarthSpecificLayer:
  def __init__(self, depth, dim, drop_path_ratio_list, heads):
    '''Basic layer of our network, contains 2 or 6 blocks'''
    self.depth = depth
    self.blocks = []

    # Construct basic blocks
    for i in range(depth):
      self.blocks.append(EarthSpecificBlock(dim, drop_path_ratio_list[i], heads))
      
  def forward(self, x, Z, H, W):
    for i in range(self.depth):
      # Roll the input every two blocks
      if i % 2 == 0:
        self.blocks[i](x, Z, H, W, roll=False)
      else:
        self.blocks[i](x, Z, H, W, roll=True)
    return x

class EarthSpecificBlock:
  def __init__(self, dim, drop_path_ratio, heads):
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    # Define the window size of the neural network 
    self.window_size = (2, 6, 12)

    # Initialize serveral operations
    self.drop_path = DropPath(drop_rate=drop_path_ratio)
    self.norm1 = LayerNorm(dim)
    self.norm2 = LayerNorm(dim)
    self.linear = MLP(dim, 0)
    self.attention = EarthAttention3D(dim, heads, 0, self.window_size)

  def forward(self, x, Z, H, W, roll):
    # Save the shortcut for skip-connection
    shortcut = x

    # Reshape input to three dimensions to calculate window attention
    reshape(x, target_shape=(x.shape[0], Z, H, W, x.shape[2]))

    # Zero-pad input if needed
    x = pad3D(x)

    # Store the shape of the input for restoration
    ori_shape = x.shape

    if roll:
      # Roll x for half of the window for 3 dimensions
      x = roll3D(x, shift=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2])
      # Generate mask of attention masks
      # If two pixels are not adjacent, then mask the attention between them
      # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
      mask = gen_mask(x)
    else:
      # e.g., zero matrix when you add mask to attention
      mask = no_mask

    # Reorganize data to calculate window attention
    x_window = reshape(x, target_shape=(x.shape[0], Z//window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], x.shape[-1]))
    x_window = TransposeDimensions(x_window, (0, 1, 3, 5, 2, 4, 6, 7))

    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
    x_window = reshape(x_window, target_shape=(-1, window_size[0]* window_size[1]*window_size[2], x.shape[-1]))

    # Apply 3D window attention with Earth-Specific bias
    x_window = self.attention(x, mask)

    # Reorganize data to original shapes
    x = reshape(x_window, target_shape=((-1, Z // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], x_window.shape[-1])))
    x = TransposeDimensions(x, (0, 1, 4, 2, 5, 3, 6, 7))

    # Reshape the tensor back to its original shape
    x = reshape(x_window, target_shape=ori_shape)

    if roll:
      # Roll x back for half of the window
      x = roll3D(x, shift=[-self.window_size[0]//2, -self.window_size[1]//2, -self.window_size[2]//2])

    # Crop the zero-padding
    x = Crop3D(x)

    # Reshape the tensor back to the input shape
    x = reshape(x, target_shape=(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]))

    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))
    return x
    
class EarthAttention3D:
  def __init__(self, dim, heads, dropout_rate, window_size):
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    # Initialize several operations
    self.linear1 = Linear(dim, dim=3, bias=True)
    self.linear2 = Linear(dim, dim)
    self.softmax = SoftMax(dim=-1)
    self.dropout = DropOut(dropout_rate)

    # Store several attributes
    self.head_number = heads
    self.dim = dim
    self.scale = (dim//heads)**-0.5
    self.window_size = window_size

    # input_shape is current shape of the self.forward function
    # You can run your code to record it, modify the code and rerun it
    # Record the number of different window types
    self.type_of_windows = (input_shape[0]//window_size[0])*(input_shape[1]//window_size[1])

    # For each type of window, we will construct a set of parameters according to the paper
    self.earth_specific_bias = ConstructTensor(shape=((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads))

    # Making these tensors to be learnable parameters
    self.earth_specific_bias = Parameters(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    TruncatedNormalInit(self.earth_specific_bias, std=0.02) 

    # Construct position index to reuse self.earth_specific_bias
    self.position_index = self._construct_index()
    
  def _construct_index(self):
    ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
    # Index in the pressure level of query matrix
    coords_zi = RangeTensor(self.window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -RangeTensor(self.window_size[0])*self.window_size[0]

    # Index in the latitude of query matrix
    coords_hi = RangeTensor(self.window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -RangeTensor(self.window_size[1])*self.window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = RangeTensor(self.window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 = Stack(MeshGrid([coords_zi, coords_hi, coords_w]))
    coords_2 = Stack(MeshGrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = Flatten(coords_1, start_dimension=1) 
    coords_flatten_2 = Flatten(coords_2, start_dimension=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = TransposeDimensions(coords, (1, 2, 0))

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += self.window_size[2] - 1
    coords[:, :, 1] *= 2 * self.window_size[2] - 1
    coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

    # Sum up the indexes in three dimensions
    self.position_index = TensorSum(coords, dim=-1)

    # Flatten the position index to facilitate further indexing
    self.position_index = Flatten(self.position_index)
    
  def forward(self, x, mask):
    # Linear layer to create query, key and value
    x = self.linear1(x)

    # Record the original shape of the input
    original_shape = x.shape

    # reshape the data to calculate multi-head attention
    qkv = reshape(x, target_shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number)) 
    query, key, value = TransposeDimensions(qkv, (2, 0, 3, 1, 4))

    # Scale the attention
    query = query * self.scale

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    attention = query @ key.T # @ denotes matrix multiplication

    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    EarthSpecificBias = self.earth_specific_bias[self.position_index]

    # Reshape the learnable bias to the same shape as the attention matrix
    EarthSpecificBias = reshape(EarthSpecificBias, target_shape=(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.type_of_windows, self.head_number))
    EarthSpecificBias = TransposeDimensions(EarthSpecificBias, (2, 3, 0, 1))
    EarthSpecificBias = reshape(EarthSpecificBias, target_shape = [1]+EarthSpecificBias.shape)

    # Add the Earth-Specific bias to the attention matrix
    attention = attention + EarthSpecificBias

    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    attention = self.mask_attention(attention, mask)
    attention = self.softmax(attention)
    attention = self.dropout(attention)

    # Calculated the tensor after spatial mixing.
    x = attention @ value.T # @ denote matrix multiplication

    # Reshape tensor to the original shape
    x = TransposeDimensions(x, (0, 2, 1))
    x = reshape(x, target_shape = original_shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x
  
class Mlp:
  def __init__(self, dim, dropout_rate):
    '''MLP layers, same as most vision transformer architectures.'''
    self.linear1 = Linear(dim, dim * 4)
    self.linear2 = Linear(dim * 4, dim)
    self.activation = GeLU()
    self.drop = DropOut(drop_rate=dropout_rate)
    
  def forward(self, x):
    x = self.linear(x)
    x = self.activation(x)
    x = self.drop(x)
    x = self.linear(x)
    x = self.drop(x)
    return x

def PerlinNoise():
  '''Generate random Perlin noise: we follow https://github.com/pvigier/perlin-numpy/ to calculate the perlin noise.'''
  # Define number of noise
  octaves = 3
  # Define the scaling factor of noise
  noise_scale = 0.2
  # Define the number of periods of noise along the axis
  period_number = 12
  # The size of an input slice
  H, W = 721, 1440
  # Scaling factor between two octaves
  persistence = 0.5
  # see https://github.com/pvigier/perlin-numpy/ for the implementation of GenerateFractalNoise (e.g., from perlin_numpy import generate_fractal_noise_3d)
  perlin_noise = noise_scale*GenerateFractalNoise((H, W), (period_number, period_number), octaves, persistence)
  return perlin_noise
