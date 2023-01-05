# Deep Learning Project - SuperTuxKart

This was a project done for UT Austin's Deep Learning Course in Fall 2022. We discuss the approach we took for this project in this write up but have ommited the code to preserve integrity of the assignment.

### Team Members: Angelo Belmonte, Kuan Yin Huang, Sheyla Corral, Soumili Kole

# Introduction
For the final project, our task is to program a SuperTuxKart ice-hockey player in a 2v2 game to play against an AI, or another player designed by either the instructors or other classmates. With our experience driving, playing similar video games, and physics, we were able to think of approaches to how we want to design our player agent’s controller. Our team approached this task by trying to let our players find the puck successfully as soon as possible and try to dribble the puck towards the opponent’s goal so that we would have a higher chance of scoring goals. We talk more about our approaches, ideas, and what we explored in the following sections of the write up.

# Agent Type Selection
The team decided on an image-based agent because the previous homeworks had been on image-based models, so we had some experience in this approach and had some boilerplate code in the previous homework solutions.

# Approach
At first, we thought that we would just need to follow the steps as in HW4 and HW5 and mainly focus on the agent’s controller, so our approach started with building a puck detection model using the model from HW5’s solution. Next, we wanted to feed the model images to train it and detect where the puck was, but we stumbled upon a problem, which was the training data. There was not available data for us to directly train, so we had to generate our own images.

To collect the image data, we first converted the puck’s true location into a local image using functions from HW4 to generate true labels for our training data. Thus, we got image coordinates of the puck location as labels. We designed the model so that it should be able to predict a puck’s coordinate if the puck is in the image; otherwise, it should predict (0, 1) as its label coordinates, signifying the puck is not in the image. Note the y-axis is flipped in pytux which is why we made the y-coordinate positive 1 in our label.

After the first model we designed and trained, we realized that the quality of the data we collect is crucial to our agent’s performance. Our first agent would stutter, get stuck at walls and inside goals, prefer power-up objects, and get stuck at the goal posts. It was obvious that our agents preferred these actions due to noise in our data. Because of this, we carefully analyzed the images we collected and the AI agents we collected through.

The team considered some AI agents available to us to collect data by running matches. Ultimately, we decided our data collection through geoffrey_agent, jurgen_agent, and yann_agent because these agents already knew how to chase after the puck, while a few others performed erratically. 

At first, we simply used a lot of images but found the data to be lopsided towards some properties, such as the puck not being in the image and the puck being very close to the agent. To solve this issue, we sectioned the images into somewhat equal parts according to whether the puck is in the image, behind the kart, near the kart, and far away from the kart. 

The model’s architecture is based on a past SuperTuxKart project we had done. At first, we based our model on HW5’s model, which was four layers of (Conv2D, Batchnorm, ReLU) blocks. Then, we tried using HW4, which was an FCN model with skip connections and contained four layers with each layer using 3 (Conv2D, Batchnorm, ReLU) blocks. Our observation between the two was that HW4 seems to work better. Although our model architecture worked on our machines, it was critiqued to be too slow even if we cut the number of layers in half. The team settled on basing our final model on HW5’s architecture.

The other part of the project is the controller design. We designed the controller so that the kart would reverse if the puck was behind it or chase after the puck. Once the kart is pushing the puck, we calculated the angle in which the kart would turn to push the puck towards the goal.

Initially, we thought designing the controller logic would be simple. Our players’ performance proved otherwise. The team not only had to produce a good model, but also we had to optimize the controller to each specific model. There were times when we used different labels on images, which meant we had to create distinct logic between them.

The team designed special cases due to unforeseen noise in the data and controller logic error. These cases were: stuck at the wall, stuck behind the goal posts, not scoring, and scoring for the enemy team. In each of these cases, we considered and tweaked the angle between the kart’s direction vector and the kart’s location to the goal post vector, the puck’s label coordinates, the kart’s location, and the kart’s x and y velocity. 

# Models
The team named our models based on famous soccer players, because at the time we made the project, the world cup was going on. The architecture of the first models we made were based on homework 5’s solution then we migrated to homework 4 solution and stayed with that until we realized that it was causing timeouts. We produced good and accurate results as shown when we visualize the model predictions. There were seldom inaccuracies, especially on far images and when the puck was not in the image. However, when we reverted back to homework 5 solution we were struggling to get a good loss that also performed well with our player logic.

## Ronaldo
Training data (1000 images):
- Agents used to collect – AI
- Labels: x, y coordinates for the puck
- Learning rate: 0.001
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Ronaldo performed as we expected. It was the first model we designed, and we made a simple controller to let it run. Ronaldo simply goes staring at the goal or the wall.

## Messi
The only difference Messi has from Ronaldo is that we used training data from different AI agents, and a better controller that would try to follow the puck.

Training data (1000 images):
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent, yoshua_agent
- Labels: x, y coordinates for the puck
- Learning rate: 0.01
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Messi performed better than Ronaldo. The agent could detect where the puck is and turns towards the puck; however, Messi sometimes dodges the puck. The agent also has trouble detecting the puck when it is not in the image.

## Neymar
Instead of training on the same model name, the team wanted an archive of models because we may have past models which would perform better.

Due to Messi’s erratic movement, the team thought the model was slow in producing predictions. Therefore, we designed a model with half of Messi’s layers.
Training data (10000 images):
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent, yoshua_agent
- Labels: x, y coordinates for the puck
- Learning rate: 0.01
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Neymar could sometimes get to the puck before the opponent agent. However, the agent performed worse than Messi.

## Giroud
Due to the subpar performance of Neymar, the team analyzed the agents we used to collect data. We found that yoshua_agent’s movements were erratic and did not actually follow the puck. We trained Girourd’s model without yoshua_agent’s collected images. The images also had a new distinction on its labels: whenever the puck was not in the image, the model should label (0, 1). 

Training data (2000 images):
- Loss: 0.06
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 0.0001
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Girourd’s performance was better than both Messi and Neymar. However, we found that Girourd would not detect the puck when it is far away. 

## Mbappe
We created Mbappe by continuing to train Girourd but with more images.

Training data (10000 images):
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 0.0001
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])


###### Observation:
Mbappe could score goals. The agent could detect where the puck is when the puck is nearby; however, the agent still had trouble when the puck was far.

## Pele
To remedy Mbappe’s issue in being unable to detect when far, we distributed our data into even parts: far, close, no puck, and behind. We also changed the no puck label to coordinate (1, 1) to help us steer out when the puck is not in view. The agent would be able to steer to this coordinate in reverse.

Training data (10000 images):
- Loss: 0.066
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 1e-3,1e-4,1e-5 about 50,50,30
- Epochs: 130
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
The team quickly decided to move away from Pele. The agent did not perform as well as Mbappe.

## Salah
Due to Pele’s underperformance, we decided to take another approach that stems from Mbappe. We reverted our no puck label coordinates back to (0, 1), and we trained our model with 10000 more images.

Training data (20000 images):
- Loss: 0.038
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 1e-4
- Epochs: 100
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Salah performed worse than Mbappe.

## Mane
The team wanted to try a different approach to improve from Mbappe. We noticed a few issues with the data that we had previously collected such as, when the puck was behind us the label was (x, -1.0) indicating that the puck was on the top edge of the image which was incorrect. We also decreased our y value when collecting images that were far away. We went from -0.2 to -0.1.


Training data (20000 images):
- Loss: 0.05
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 1e-3 (worked okay for a few epochs),1e-4 (worked best for 50 epochs)
- Epochs: 50
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Mane did a lot better than Salah and could smoothly turn and follow the puck; however, the team was running out of time. We decided to use Mbappe as our final model.

## Ronaldinho
Mbappe’s submission actually did not pass the allotted time per step. To adjust to a faster player, we used homework 5’s model because it contains a simpler architecture. Homework 5 did not have up and down blocks which homework 4’s model architecture had. It only had a single convolutional block per layer.

Training data (20000 images):
- Loss: 0.107
- Agents used to collect – AI, geoffrey_agent, jurgen_agent, yann_agent
- Labels: x, y coordinates for the puck
- Learning rate: 0.01
- Epochs: 100
- Transform = Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])

###### Observation:
Ronaldinho passed the time constraint. We used 16, 32, 32, 32 layers to design our model, and the same images we trained Mane. The previous models had good loss; however, they took a long time to detect each frame which was causing a time out. We had two different Ronaldinho models, one with loss 0.107 and one with loss 0.076. Looking at the videos using these models it seemed that they weren’t performing nearly as well as the Mbappe model but we were still able to score goals with them. Ultimately, we stuck with the one with higher loss as it worked the best with our player.

## Image Statistics:
- 2387 far
- 2632 close
- 3511 no_puck (0,1)
- 2719 behind
- 26008 total

# Predictions
The team used a function to visualize our model’s prediction against the puck label on the image. Our models correctly predicted if the puck was in the image. In the case where the puck is not in the image, we designed the model to predict a certain coordinate which we used on our controller to on special cases. However, due to the noise from walls and power up boxes, our model sometimes produced incorrect predictions when the puck was not in the image.

Green: Model prediction
Red: Image label
<img width="1500" alt="Picture1" src="https://user-images.githubusercontent.com/7693425/206937943-1dbb7273-faf1-4dd7-9c84-8b2d92fe27b7.png">

<img width="1500" alt="Picture2" src="https://user-images.githubusercontent.com/7693425/206937946-0f024b9e-2081-4317-b707-be0da2f93ddb.png">



# Controller
##Speed and Acceleration Logic:
Some of our initial observations:
- The agent would pass the puck when readjusting its angle towards the puck. 
- The enemy would steal the puck if our agent was too slow with the puck. 
- Our agent would fail to gain possession if our agent was too slow away from the puck.

Due to our observations, we set the agent to slow down when readjusting its angle towards the puck, speed up when the agent has the puck set in front and up close, and turn on the nitros if the agent sees the puck in the image, but far away.
## Goal Logic:
The team discovered how to get the kart’s vector direction by using the kart’s location and front vector. We also discovered the numpy arccos function to find the angle between two vectors. By taking the goal’s coordinate and kart’s location coordinates, we found the two vectors to use the arccos function on. Taking the dot product of the two vectors, then applying the arccos function returns an angle between 0 to pi. We also applied the numpy cross function to check whether the angle from kart’s vector direction to goal direction is clockwise or counter clockwise. After finding the angle, we produced cases that would adjust the kart’s aim point accordingly–to push into the enemy goal and push away from our goal.

An example of a case we considered was based on numpy arccos return value–which is between 0 to 3.14. If the player kart were near the enemy goal and the angle returned was less than 1.57 which is half of pi, then the kart’s aim point should be on the puck’s opposite side from the enemy goal. If the kart were on its own goal, then we applied the reverse logic to push the puck away from its goal.
## Avoiding Getting Stuck Logic:
We found our kart to sometimes get stuck either in goals (both the opponent’s and our own) or in front of walls. Thus, we added code in the controller to tell the kart what to do in case it gets stuck in those situations.

To avoid getting stuck in goals, we used the 3D coordinates of the ice-hockey field to implement this. We know that the goal line y-coordinate is either -64.5 or 64.5, so we used this along with the kart’s current location to determine whether the kart is inside the goal. If the kart was inside the goal, then we would also determine if the kart was in the right/left half of the field. Using this information, we could tell the kart to back up while steering in the right direction to enter the field again. We later found out that there was a flaw in this logic, since the kart would still back up if its coordinates indicated that it was beyond the goal line but it was actually facing the field and not the goal. This resulted in the kart always backing up and getting stuck in the goal. Thus, we later decided to also use the ‘front’ player state to help us determine where the kart was facing by defining: kart_location = kart[‘front’] - kart[‘location’]. So, our new logic also checked where the kart was facing to determine if the kart should back up or not. If it was facing the goal, then it would need to back up, but if it was facing the field, then it would not trigger this part of the code and act how it normally would be.

As for avoiding getting stuck in front of walls, we used the kart’s velocity in both x and y directions, the angle of the kart and the midpoint of the goal line, and also the kart’s direction that was mentioned in the last paragraph. We found out that when the kart gets stuck on the wall, its x and y velocity would drop significantly to lower than 0.2, and its z coordinate direction would not be equal to 0. Thus, we used this information to detect whether the kart was stuck on the wall or not. Then, we would have to also know where the kart was stuck on. We used the kart’s direction to determine which wall the kart was stuck on. If the direction of the kart mainly pointed towards the x axis, then we would know that it got stuck on the left/right walls. Otherwise, it got stuck on the top/bottom walls. Then, depending on which team our player was in (RED/BLUE) and which wall it got stuck on, we designed different ways for the kart to back up, so that it would face the field and the opponent’s goal to let it find the puck again.
# Conclusion
The team found the project a good learning experience. We had to figure out how to collect data and how to set an agent to play a game with AI. By designing a strategy of following the puck and tweaking its steering to the enemy’s goal posts when it has possession of the puck, our agent could score a few points against the other agents. 

After several improvements to our controller and model, our agent continues to struggle backing up when it hits the wall, and determining the correct movement to score a goal. We think we could improve our agent’s strategy further; however, by the time we figured out we had a good model, the team did not have enough time to tune our controller to best fit the latest model. Ultimately, the team believes that our model predicts well as shown when we visualize the model predictions. 

If we had more time, we feel that we could get a better model which would not seldom predict incorrectly by further lessening its loss. Then, our controller logic would be simpler to design and optimize. Lastly, the team also wants to experience designing a state based agent to see how they work compared to an image based agent.
