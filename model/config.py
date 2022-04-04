
# RCVNet, VNet
params = {'in_channels': 1,  # Input channel of first layer (modified in model for later layers)
          'out_channels': 16,  # Output channel of first layer (modified in model for later layers)
          'create_layer_1': False,  # Internal param to indicate layer 1 is being created
          'create_layer_2': False,  # Internal param to indicate layer 2 is being created
          'kernel_size': (5, 5, 5),  # kernel size
          'input_shape': (256, 256, 256),  # input volume shape l x b x w
          'patch_size': (128, 128, 128),  # patch size
          'num_classes': 79,  # number of classes
          'out': False,  # Internal param to indicate output layer creation
          'input': True,  # Internal param to make ip and op channels to 1st layer equal for
          # residuals to be used (1 -> 16)
          'epochs': 800,  # number of training epochs
          'lr': 0.001,  # Initial learning rate
          'milestones': [320, 640, 720],  # milestones for learning rate decay torch scheduler
          'model_name': 'RCVNet',  # Choices between [RCVNet, CropNet3d, MultiResVNet]. Currently name
          # is same as python file for model class
          'sub_model_name': 'vnet',  # Choices between [vnet, vnet_2d_3d, vnet_asym, vnet_sym,
          # vnet_denseadd, vnet_exclusion, vnet_se, attention, multiloss]
          # multiloss only with MultiResVNet
          'ml_alpha': 0.5,  # The alpha value for multiloss training of MultiResVNet
          'checkpoint_dir': 'checkpoints',  # Main checkpoint director
          'experiment_id': 'Example',  # Experiment ID to store models
          'pretrained': False,
          'gen_random': True,  # Internal param to make model generate checkpoints, None during eval
          'gpu_map': {}  # Maps model blocks by key (python layer var name) to different gpus
          # Eg. 'decoder_block_2': 'cuda:1', 'decoder_block_1': 'cuda:1',
          # 'output_block': 'cuda:1'} # MUST ALWAYS map blocks after a mapping sequence ends
          }

"""
For other RCVNET type models, just change the sub_model_name to the following while keeping everything else above intact:
'vnet_2d_3d', 'vnet_denseadd', 'vnet_attention', 'vnet_excusion', 'vnet_se', 'vnet_attention'

for 'vnet_sym', 'vnet_asym' also change kernel size to (3,3,3)

During evaluation of of RCVNet with attention, please change the class_string to 'RCVNetAttenion' as attention uses a 
different class due to minor changes in macro architecture

PLEASE CHANGE THE experiment_id FOR EVERY EXPERIMENT, OTHERWISE OLD EXPERIMENTS WILL BE OVERWRITTEN BY NEW EXPERIMENTS.


FOR CropNet3D and MultiResVNet, keep the above params dictionary and change the sub_model_type to vnet (and multiloss 
only for MultiResVNet. During eval, please be sure to refer to the correct class and file in class_string and 
file_string.

For example, MultiResVNet with multiloss, 
file_string='MultiResVNet', 
class_string='MultiResVNetML'
"""

