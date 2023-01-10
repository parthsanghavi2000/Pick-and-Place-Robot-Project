import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1,7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    
    velocity = np.vstack((v_in, omega_in))
    Jacobian_vel = calcJacobian(q_in)
    x=[]
    for i in range(0,6):
    	if(np.isnan(velocity[i])):
    		x.append(i)
    Jacobian_vel = np.delete(Jacobian_vel, x, axis =0)
    velocity = np.delete(velocity,x,axis=0)
    if(np.linalg.det(Jacobian_vel@Jacobian_vel.T) <= 1):
    	dq = np.linalg.lstsq(Jacobian_vel,velocity,rcond=None)
    	dq = dq[0]
    	dq = dq.reshape(7,)
    	
    else:
    	dq= np.linalg.pinv(Jacobian_vel)@velocity
    	dq = dq.reshape(7,)	

    return dq
