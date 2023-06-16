# Project goals

The goal of this project is to train an agent to play at Atari Breakout game.

We used  the **MinAtar Breakout-v1** environment of the gym library and developed two types of agent : a DQN and a Double DQN agent. 

# Structure of the project
The project is composed of :
* Two *Jupyter Notebooks* : one for the agent's training (`train_agent.ipynb`) and another for the explainability (`Explainability.ipynb`).
* A folder `agents` composed of the two classes of agents.
* A folder `networks_weights` where the weights of the agents are saved.
* A folder `games` where the gif files of the games played by agents are saved.
* Two scripts (`train_agent.py` and `play_game.py`) for training and play games by specifying arguments. 


# Docker

We can use a Docker image to execute the two scripts. This image is based on the `Dockerfile` which pulls the latest pytorch image available on the Docker Hub and installs the necessary libraries (*gym*, *minatar*...).

For building this image : `docker build -t rl-breakout-image .`

## Execution of the training 
We can train an agent in a Docker container by running this command :   
```docker run --rm -v ${PWD}/games/:/app/games -v ${PWD}/networks_weights/:/app/networks_weights rl-breakout-image train_agent.py ARGS```  
where **ARGS** are the arguments we can use in the script.

The arguments are : 
* `--agent` : the agent we train (**dqn** or **ddqn**) - by default **ddqn**.
* `--gamma`: by default **0.99**.
* `--lr`: the learning rate - by default **1e-3**.
* `--batch_size`: by default **32**.
* `--epsilon_min`: by default **1e-2**.
* `--epsilon_decay`: by default **0.999**.
* `--memory_size`: by default **100 000**.
* `--max_steps`: by default **5000**.
* `--play_game`: boolean to choose if we play a game after the training - by default **False**.

For example, we can execute :  
```docker run --rm -v ${PWD}/networks_weights/:/app/networks_weights -v ${PWD}/games/:/app/games rl-breakout-image train_agent.py --agent 'dqn' --nb_episode 1000 --play_game True```

**Notice: the "game" volume is unnecessary if we don't play a game after the training.**


## Play a game
We can make the agent play and save the game as a gif by running the following command:  
```docker run --rm -v ${PWD}/games/:/app/games rl-breakout-image play_game.py ARGS```

The arguments are :
* `--agent` : the agent we use (**dqn** or **ddqn**) : the best weights of the agent will be loaded - by default **ddqn** weights.
* `--path_weights`: path to specific agent weights - by default **None**.
* `--gif_name`: by default **game.gif**.


For example :  
```docker run --rm -v ${PWD}/games/:/app/games rl-breakout-image play_game.py --agent 'dqn' --gif_name 'dqn-agent-game1.gif'```
