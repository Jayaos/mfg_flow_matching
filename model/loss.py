import torch


def vf_loss_fn(vector_field, x_trajectory, timesteps, timestep_size):
    """
    x_trajectory: (batch_size, len(timesteps), dim)
    """
    velocity_target = x_trajectory[:,1:,] - x_trajectory[:,:-1,] # (batch_size, len(timesteps)-1, dim)
    velocity_forward_pred = vector_field(timesteps[:-1], x_trajectory[:,:-1,:])*timestep_size
    loss = (velocity_forward_pred - velocity_target).pow(2).mean()
    
    return {"loss" : loss}

def image_vf_loss_fn(vector_field, x_trajectory, timesteps, timestep_size):
    """
    x_trajectory: (batch_size, len(timesteps), channel, w, h)
    """
    batch_size = x_trajectory.shape[0]
    velocity_target = x_trajectory[:,1:,] - x_trajectory[:,:-1,] # (batch_size, len(timesteps)-1, c, w, h)
    velocity_target = velocity_target.flatten(0,1) # (batch_size*(len(timesteps)-1), c, w, h)
    velocity_forward_pred = vector_field(timesteps[:-1].unsqueeze(1).repeat(1, batch_size).flatten(0, 1), 
                                         x_trajectory[:,:-1,:].flatten(0,1))*timestep_size

    loss = (velocity_forward_pred - velocity_target).pow(2).mean()

    return {"loss" : loss}


def classifier_loss_fn(classifier, x_p, x_q):

    logit_x = classifier(x_p)
    logit_q = -classifier(x_q)
    loss_x = torch.nn.functional.softplus(logit_x, beta=1).mean()
    loss_q = torch.nn.functional.softplus(logit_q, beta=1).mean()
    
    return {"loss" : loss_x + loss_q, "loss_x" : loss_x, "loss_q" : loss_q}


def particle_optimization_loss_fn(particle_0, 
                                  particle_trajectory, 
                                  classifier, 
                                  kinetic_loss_weight=1.0, 
                                  classifier_loss_weight=1.0):

    # kinetic loss
    init_diff = particle_trajectory[0] - particle_0 # (1, batch_size, dims)
    trajectory_diff = particle_trajectory[1:] - particle_trajectory[:-1] # (num_ts-2, batch_size, dim)
    # sum over timesteps and data dims
    # avg over batch
    kinetic_loss = (trajectory_diff**2).sum(dim=list(range(2, particle_trajectory.ndim))).sum(dim=[0]).mean()
    # sum over data dims and avg over batch
    kinetic_loss += (init_diff**2).sum(dim=list(range(1, particle_trajectory.ndim))).mean()
    kinetic_loss /= 2

    # classifier loss
    classifier_loss = -classifier(particle_trajectory[-1]).mean()

    return {"loss" : kinetic_loss_weight*kinetic_loss + classifier_loss_weight*classifier_loss, 
            "kinetic_loss" : kinetic_loss, 
            "classifier_loss" : classifier_loss}
