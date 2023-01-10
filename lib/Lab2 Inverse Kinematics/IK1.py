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

	

	def __init__(self, linear_tol=1e-3, angular_tol=1e-6, max_steps=10000, min_step_size=1e-5):
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

		# get joint limits
		lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
		upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

		flag = False
		rate = 5e-1
		while True:
			current	=  FK().forward(q)[1]
			tar = target[0:3,-1]
			cur = current[0:3,-1]
			disp = tar - cur	  
			R  =	 np.dot(np.linalg.inv(current[0:3,0:3]),target[0:3,0:3])
			s = (R-R.T)/2
			a = np.array([s[2,1], s[0,2], s[1,0]])
			axis = current[0:3,0:3] @ a
			
			current	=  FK().forward(q)[1]
			v = disp
			omega = axis
		   
			dq_P= IK_velocity(q,v,omega)
			
			# Secondary Task - Center Joints
			offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
			dq_jct = rate * - offset

			J = calcJacobian(q)
			Z = null_space(J).flatten()
			
			dq_C = np.dot(dq_jct, Z) * (Z / np.linalg.norm(Z)**np.linalg.norm(Z))

			dq = 0.5 * (dq_P + 2 * dq_C)
			num += 1
			
			if num > self.max_steps or np.linalg.norm(dq) <= self.min_step_size:
				break  
			q = q + dq
	
		print("steps")
		print(num)
		



		return q,  rollout
		
if __name__ == "__main__":

	np.set_printoptions(suppress=True, precision=5)

	ik = IK()

	# matches figure in the handout
	seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
	x = np.array([pi, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
	joints1, target	=  ik.fk.forward(x)
   
	q, rollout = ik.gdik(target, seed)

	for i, q in enumerate(rollout):
		joints, pose = ik.fk.forward(q)	 # joint_positions & T0e
		

	
	print("error: ", (x-q))
	print("Iterations:", len(rollout))	

