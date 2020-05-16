from CNN import obs_to_torch
from mainEXP import load_model
from gym_pcgrl import wrappers
import matplotlib.pyplot as plt
import torch
import gym
import numpy as np
import time


if torch.cuda.is_available():
    print('gpu found')
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def reshape_obs(obs):
    return obs[:,np.newaxis,:,:,:]


def build(game, representation, model_path, make_gif, gif_name, **kwargs):

    env_name = '{}-{}-v0'.format(game, representation)

    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    crop_size = kwargs.get('cropped_size',28)

    env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size,**kwargs)

    n_actions = env.action_space.n
    obs = env.reset()
    # print(obs.shape)


    model, _,_ = load_model(model_path, obs.shape[-1], crop_size, n_actions)
    #obs = reshape_obs(obs)
    # print(obs.shape)

    frames = []

    done = False
    while not done:
        if make_gif:
            frames.append(env.render(mode='rgb_array'))
        env.render()
        action = model.forward(obs_to_torch(obs))
            # print(actions)
        obs, reward, done, info = env.step(action)
        #obs = reshape_obs(obs)
    print(info)

    print(len(frames))
    if make_gif:
        frames[0].save(gif_name,save_all=True,append_images = frames[1:])

    time.sleep(10)



################################## MAIN ########################################
game = 'binary'
representation = 'narrow'
make_gif = True
kwargs = {
            'change_percentage': 0.4,
            'verbose': True,
            'render': True,
}

#model_path = 'models/{}/{}'.format(game,representation)
model_path = 'runs/'
gif_name = 'gifs/{}-{}.gif'.format(game, representation)
print(gif_name)


if __name__ == '__main__':
    build(game, representation, model_path, make_gif, gif_name, **kwargs)
