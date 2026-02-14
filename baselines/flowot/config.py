

class FlowOTHDGaussianConfig():

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
                 wasserstein_loss_weight,
                 num_iter,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 checkpoint_step,
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
        self.wasserstein_loss_weight = wasserstein_loss_weight
        self.num_iter = num_iter
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.checkpoint_step = checkpoint_step
        self.saving_dir = saving_dir


class FlowOTImageConfig():

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
                 velocity_field_training_step,
                 wasserstein_loss_weight,
                 num_iter,
                 ode_solver,
                 odeint_batch_size,
                 num_timesteps,
                 vae_config_dir,
                 vae_model_dir,
                 checkpoint_step,
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
        self.velocity_field_training_step = velocity_field_training_step
        self.wasserstein_loss_weight = wasserstein_loss_weight
        self.num_iter = num_iter
        self.ode_solver = ode_solver
        self.odeint_batch_size = odeint_batch_size
        self.num_timesteps = num_timesteps
        self.vae_config_dir = vae_config_dir
        self.vae_model_dir = vae_model_dir
        self.checkpoint_step = checkpoint_step
        self.saving_dir = saving_dir


class ShoebagsDatasetConfig():

    def __init__(self, 
                 shoes_dataset_dir,
                 bags_dataset_dir,
                 shoes_latent_params_dir,
                 bags_latent_params_dir,
                 train_ratio,
                 seed
                 ):
        
        self.shoes_dataset_dir = shoes_dataset_dir
        self.bags_dataset_dir = bags_dataset_dir
        self.shoes_latent_params_dir = shoes_latent_params_dir
        self.bags_latent_params_dir = bags_latent_params_dir
        self.train_ratio = train_ratio
        self.seed = seed


class CelebADatasetConfig():

    def __init__(self, 
                 data_dir,
                 train_male_latent_params_dir,
                 train_female_latent_params_dir,
                 valid_male_latent_params_dir,
                 valid_female_latent_params_dir,
                 ):
        
        self.data_dir = data_dir
        self.train_male_latent_params_dir = train_male_latent_params_dir
        self.train_female_latent_params_dir = train_female_latent_params_dir
        self.valid_male_latent_params_dir = valid_male_latent_params_dir
        self.valid_female_latent_params_dir = valid_female_latent_params_dir