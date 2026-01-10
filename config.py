class MFGFlowToyExampleConfig():

    def __init__(self, 
                 classifier_hidden_dims,
                 classifier_activation,
                 classifier_learning_rate,
                 classifier_training_batch_size,
                 initial_classifier_training_step,
                 classifier_intermediate_training_frequency,
                 intermediate_classifier_training_step,
                 velocity_field_hidden_dims,
                 velocity_field_layer_type,
                 velocity_field_activation,
                 velocity_field_initialization,
                 velocity_field_initialization_training_step,
                 velocity_field_learning_rate,
                 velocity_field_training_batch_size,
                 initial_velocity_field_training_step,
                 velocity_field_training_step,
                 initial_particle_optimization_epoch,
                 particle_optimization_epoch,
                 particle_optimization_learning_rate,
                 particle_optimization_batch_size,
                 kinetic_loss_weight,
                 classifier_loss_weight,
                 epochs,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 seed,
                 saving_dir
                 ):
        
        self.classifier_hidden_dims = classifier_hidden_dims
        self.classifier_activation = classifier_activation
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_training_batch_size = classifier_training_batch_size
        self.initial_classifier_training_step = initial_classifier_training_step
        self.classifier_intermediate_training_frequency = classifier_intermediate_training_frequency
        self.intermediate_classifier_training_step = intermediate_classifier_training_step
        self.velocity_field_hidden_dims = velocity_field_hidden_dims
        self.velocity_field_layer_type = velocity_field_layer_type
        self.velocity_field_activation = velocity_field_activation
        self.velocity_field_initialization = velocity_field_initialization
        self.velocity_field_initialization_training_step = velocity_field_initialization_training_step
        self.velocity_field_learning_rate = velocity_field_learning_rate
        self.velocity_field_training_batch_size = velocity_field_training_batch_size
        self.initial_velocity_field_training_step = initial_velocity_field_training_step
        self.velocity_field_training_step = velocity_field_training_step
        self.initial_particle_optimization_epoch = initial_particle_optimization_epoch
        self.particle_optimization_epoch = particle_optimization_epoch
        self.particle_optimization_learning_rate = particle_optimization_learning_rate
        self.particle_optimization_batch_size = particle_optimization_batch_size
        self.kinetic_loss_weight = kinetic_loss_weight
        self.classifier_loss_weight = classifier_loss_weight
        self.epochs = epochs
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.seed = seed
        self.saving_dir = saving_dir


class MFGFlowHDGaussianConfig():

    def __init__(self, 
                 dim,
                 classifier_hidden_dims,
                 classifier_activation,
                 classifier_learning_rate,
                 classifier_training_batch_size,
                 classifier_initial_training_step,
                 classifier_intermediate_training_frequency,
                 classifier_intermediate_training_step,
                 velocity_field_hidden_dims,
                 velocity_field_layer_type,
                 velocity_field_activation,
                 velocity_field_learning_rate,
                 velocity_field_training_batch_size,
                 velocity_field_initialization,
                 velocity_field_initialization_training_step,
                 velocity_field_training_step,
                 particle_optimization_training_epoch,
                 particle_optimization_learning_rate,
                 particle_optimization_batch_size,
                 kinetic_loss_weight,
                 classifier_loss_weight,
                 epochs,
                 epoch_data_size,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 saving_dir,
                 ):
        
        self.dim = dim
        self.classifier_hidden_dims = classifier_hidden_dims
        self.classifier_activation = classifier_activation
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_training_batch_size = classifier_training_batch_size
        self.classifier_initial_training_step = classifier_initial_training_step
        self.classifier_intermediate_training_frequency = classifier_intermediate_training_frequency
        self.classifier_intermediate_training_step = classifier_intermediate_training_step
        self.velocity_field_hidden_dims = velocity_field_hidden_dims
        self.velocity_field_layer_type = velocity_field_layer_type
        self.velocity_field_activation = velocity_field_activation
        self.velocity_field_learning_rate = velocity_field_learning_rate
        self.velocity_field_training_batch_size = velocity_field_training_batch_size
        self.velocity_field_initialization = velocity_field_initialization
        self.velocity_field_initialization_training_step = velocity_field_initialization_training_step        
        self.velocity_field_training_step = velocity_field_training_step
        self.particle_optimization_training_epoch = particle_optimization_training_epoch
        self.particle_optimization_learning_rate = particle_optimization_learning_rate
        self.particle_optimization_batch_size = particle_optimization_batch_size
        self.kinetic_loss_weight = kinetic_loss_weight
        self.classifier_loss_weight = classifier_loss_weight
        self.epochs = epochs
        self.epoch_data_size = epoch_data_size
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.saving_dir = saving_dir


class MFGFlowOTCelebAConfig():

    def __init__(self, 
                 classifier_learning_rate,
                 classifier_training_batch_size,
                 classifier_initial_training_step,
                 classifier_intermediate_training_frequency,
                 classifier_intermediate_training_step,
                 velocity_field_encoding_dims,
                 velocity_field_decoding_dims,
                 velocity_field_kernel_sizes,
                 velocity_field_strides,
                 velocity_field_initialization,
                 velocity_field_initialization_training_step,
                 velocity_field_learning_rate,
                 velocity_field_training_batch_size,
                 initial_velocity_field_training_step,
                 velocity_field_training_step,
                 initial_particle_optimization_epoch,
                 particle_optimization_epoch,
                 particle_optimization_learning_rate,
                 particle_optimization_batch_size,
                 kinetic_loss_weight,
                 classifier_loss_weight,
                 epochs,
                 epoch_training_size,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 which_benchmark,
                 saving_dir,
                 ):

        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_training_batch_size = classifier_training_batch_size
        self.classifier_initial_training_step = classifier_initial_training_step
        self.classifier_intermediate_training_frequency = classifier_intermediate_training_frequency
        self.classifier_intermediate_training_step = classifier_intermediate_training_step
        self.velocity_field_encoding_dims = velocity_field_encoding_dims
        self.velocity_field_decoding_dims = velocity_field_decoding_dims
        self.velocity_field_kernel_sizes = velocity_field_kernel_sizes
        self.velocity_field_strides = velocity_field_strides
        self.velocity_field_initialization = velocity_field_initialization
        self.velocity_field_initialization_training_step = velocity_field_initialization_training_step
        self.velocity_field_learning_rate = velocity_field_learning_rate
        self.velocity_field_training_batch_size = velocity_field_training_batch_size
        self.initial_velocity_field_training_step = initial_velocity_field_training_step
        self.velocity_field_training_step = velocity_field_training_step
        self.initial_particle_optimization_epoch = initial_particle_optimization_epoch
        self.particle_optimization_epoch = particle_optimization_epoch
        self.particle_optimization_learning_rate = particle_optimization_learning_rate
        self.particle_optimization_batch_size = particle_optimization_batch_size
        self.kinetic_loss_weight = kinetic_loss_weight
        self.classifier_loss_weight = classifier_loss_weight
        self.epochs = epochs
        self.epoch_training_size = epoch_training_size
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.which_benchmark = which_benchmark
        self.saving_dir = saving_dir


class ToyDatasetConfig():

    def __init__(self, 
                 data_name,
                 img_dir,
                 training_size,
                 test_size,
                 ):
        
        self.data_name = data_name
        self.img_dir = img_dir
        self.training_size = training_size
        self.test_size = test_size


class MFGFlowImageConfig():

    def __init__(self, 
                 classifier_channels,
                 classifier_use_bias,
                 classifier_learning_rate,
                 classifier_training_batch_size,
                 initial_classifier_training_step,
                 classifier_intermediate_training_frequency,
                 intermediate_classifier_training_step,
                 velocity_field_encoding_dims,
                 velocity_field_decoding_dims,
                 velocity_field_kernel_sizes,
                 velocity_field_strides,
                 velocity_field_initialization,
                 velocity_field_initialization_training_step,
                 velocity_field_learning_rate,
                 velocity_field_training_batch_size,
                 initial_velocity_field_training_step,
                 velocity_field_training_step,
                 initial_particle_optimization_epoch,
                 particle_optimization_epoch,
                 particle_optimization_learning_rate,
                 particle_optimization_batch_size,
                 kinetic_loss_weight,
                 classifier_loss_weight,
                 epochs,
                 epoch_training_ratio,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 vae_config_dir,
                 vae_model_dir,
                 saving_dir,
                 ):
        
        self.classifier_channels = classifier_channels
        self.classifier_use_bias = classifier_use_bias
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_training_batch_size = classifier_training_batch_size
        self.initial_classifier_training_step = initial_classifier_training_step
        self.classifier_intermediate_training_frequency = classifier_intermediate_training_frequency
        self.intermediate_classifier_training_step = intermediate_classifier_training_step
        self.velocity_field_encoding_dims = velocity_field_encoding_dims
        self.velocity_field_decoding_dims = velocity_field_decoding_dims
        self.velocity_field_kernel_sizes = velocity_field_kernel_sizes
        self.velocity_field_strides = velocity_field_strides
        self.velocity_field_initialization = velocity_field_initialization
        self.velocity_field_initialization_training_step = velocity_field_initialization_training_step
        self.velocity_field_learning_rate = velocity_field_learning_rate
        self.velocity_field_training_batch_size = velocity_field_training_batch_size
        self.initial_velocity_field_training_step = initial_velocity_field_training_step
        self.velocity_field_training_step = velocity_field_training_step
        self.initial_particle_optimization_epoch = initial_particle_optimization_epoch
        self.particle_optimization_epoch = particle_optimization_epoch
        self.particle_optimization_learning_rate = particle_optimization_learning_rate
        self.particle_optimization_batch_size = particle_optimization_batch_size
        self.kinetic_loss_weight = kinetic_loss_weight
        self.classifier_loss_weight = classifier_loss_weight
        self.epochs = epochs
        self.epoch_training_ratio = epoch_training_ratio
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.vae_config_dir = vae_config_dir
        self.vae_model_dir = vae_model_dir
        self.saving_dir = saving_dir


class ShoebagsDatasetConfig():

    def __init__(self, 
                 shoes_dataset_dir,
                 bags_dataset_dir,
                 training_shoes_latent_params_dir,
                 training_bags_latent_params_dir,
                 test_shoes_latent_params_dir,
                 test_bags_latent_params_dir,
                 train_ratio,
                 seed
                 ):
        
        self.shoes_dataset_dir = shoes_dataset_dir
        self.bags_dataset_dir = bags_dataset_dir
        self.training_shoes_latent_params_dir = training_shoes_latent_params_dir
        self.training_bags_latent_params_dir = training_bags_latent_params_dir
        self.test_shoes_latent_params_dir = test_shoes_latent_params_dir
        self.test_bags_latent_params_dir = test_bags_latent_params_dir
        self.train_ratio = train_ratio
        self.seed = seed


class CelebADatasetConfig():

    def __init__(self, 
                 data_dir,
                 training_male_latent_params_dir,
                 training_female_latent_params_dir,
                 test_male_latent_params_dir,
                 test_female_latent_params_dir
                 ):
        
        self.data_dir = data_dir
        self.training_male_latent_params_dir = training_male_latent_params_dir
        self.training_female_latent_params_dir = training_female_latent_params_dir
        self.test_male_latent_params_dir = test_male_latent_params_dir
        self.test_female_latent_params_dir = test_female_latent_params_dir


class BaselineImageConfig():

    def __init__(self, 
                 baseline,
                 velocity_field_encoding_dims,
                 velocity_field_decoding_dims,
                 velocity_field_kernel_sizes,
                 velocity_field_strides,
                 learning_rate,
                 training_batch_size,
                 max_training_step,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 vae_config_dir,
                 vae_model_dir,
                 checkpoint,
                 saving_dir,
                 ):
        
        self.baseline = baseline
        self.velocity_field_encoding_dims = velocity_field_encoding_dims
        self.velocity_field_decoding_dims = velocity_field_decoding_dims
        self.velocity_field_kernel_sizes = velocity_field_kernel_sizes
        self.velocity_field_strides = velocity_field_strides
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.max_training_step = max_training_step
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.vae_config_dir = vae_config_dir
        self.vae_model_dir = vae_model_dir
        self.checkpoint = checkpoint
        self.saving_dir = saving_dir


class BaselineHDGaussianConfig():

    def __init__(self,
                 dim,
                 baseline,
                 velocity_field_hidden_dims,
                 velocity_field_layer_type,
                 velocity_field_activation,
                 learning_rate,
                 training_batch_size,
                 max_training_step,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 checkpoint,
                 saving_dir,
                 ):
        
        self.dim = dim
        self.baseline = baseline
        self.velocity_field_hidden_dims = velocity_field_hidden_dims
        self.velocity_field_layer_type = velocity_field_layer_type
        self.velocity_field_activation = velocity_field_activation
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.max_training_step = max_training_step
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.checkpoint = checkpoint
        self.saving_dir = saving_dir


class BaselineOTCelebAConfig():

    def __init__(self, 
                 baseline,
                 velocity_field_encoding_dims,
                 velocity_field_decoding_dims,
                 velocity_field_kernel_sizes,
                 velocity_field_strides,
                 learning_rate,
                 training_batch_size,
                 max_training_step,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 checkpoint,
                 which_benchmark,
                 saving_dir,
                 ):

        self.baseline = baseline
        self.velocity_field_encoding_dims = velocity_field_encoding_dims
        self.velocity_field_decoding_dims = velocity_field_decoding_dims
        self.velocity_field_kernel_sizes = velocity_field_kernel_sizes
        self.velocity_field_strides = velocity_field_strides
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.max_training_step = max_training_step
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.which_benchmark = which_benchmark
        self.checkpoint = checkpoint
        self.saving_dir = saving_dir
