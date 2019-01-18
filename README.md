# Reinforcement Learning with Frozen Lake Game Implementation
This is a playable game derived from the known "Frozen Lake" game by Open AI Gym. It is written in Python and playable via the console.
I put it into a docker because the game is not executable with a Windows machine due to a specific module.
I also put in two files that implement **Reinforcement Learning** step by step. Q-Learning with a table and a Q-Neural Network.
You can run them via the terminal or simply copy the code into a jupyter cell and execute it.

## Playing the game
If you have a linux OS, you can just download the **game.py** and **util folder** and play the game. Otherwise, download the whole repo and follow the next steps.
Open a terminal in the folder where game.py is and run **python game.py**. Enjoy! :)

## Starting the docker container
If you want the whole docker environment, you need to do the following:

1. cd into directory with dockerfile
2. Build image from dockerfile
   $ docker build -t frozenlake -f ./dockerfile .
3. Run container from image
	 $ docker run -d -p 8888:8888 frozenlake 
4. Get token for first login (go to 5. and 6. if nothing shows)
   $ jupyter notebook list
5. Get container ID
	 $ docker container ls -a
6. Find token and url for jupyter
	 $ docker logs <containerID>
