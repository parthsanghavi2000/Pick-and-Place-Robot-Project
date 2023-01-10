import numpy as np
from math import *
from scipy.linalg import null_space	 # returns an orthonormal basis!


#Calling the functions from the lib folder
from lib.calculateFK import FK
from lib.calcJacobian import calcJacobian
from lib.IK_velocity import IK_velocity

# Writing the code for solving Inverse Kinematics

class IK:
	# Upper and Lower Joint Limits
	lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
	upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

	center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint

	fk = FK()

	

	def __init__(self, linear_tol=1e-3, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
		# Setting up the parameters for the Gradient descent IK solver
		self.linear_tol = linear_tol 
		self.angular_tol = angular_tol
		self.max_steps = max_steps
		self.min_step_size = min_step_size


	#Inverse Kinematics solver
	def inverse(self, target, seed):
		#Initialting the variables 
		q = seed
		rollout = []
		# counter 
		num = 0	 
		flag = False
		rate = 7e-1
		while True:
			current	=  FK().forward(q)[1]
			tar = target[0:3,-1]
			cur = current[0:3,-1]
			disp = tar - cur	  
			R  = np.dot(current[0:3,0:3].T,target[0:3,0:3])
			s = (R-R.T)*(0.5)
			a = np.array([s[2,1], s[0,2], s[1,0]])
			axis = current[0:3,0:3] @ a
			v = disp
			omega = axis		   
			dq_P, J = IK_velocity(q,v,omega,ret=True)
			
			# Secondary Task - Center Joints
			offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
			dq_jct = rate * -offset

			Z = null_space(J).flatten()
			
			dq_C = np.dot(dq_jct, Z) * (Z / np.linalg.norm(Z)*np.linalg.norm(Z))

			dq = 0.5 * (dq_P) +  dq_C
			num += 1
			
			if num > self.max_steps or np.linalg.norm(dq) <= self.min_step_size:
				break  
			q = q + dq
	

		return q
