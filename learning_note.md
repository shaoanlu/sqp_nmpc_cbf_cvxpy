## Learning Note

### Linear Control Barrier Function
- Constructing CBF constraint on every point of the predicted state `x_traj` make it result more aggressive around the obstacle
- There will be a deadlock if the linearized cbf line if perpendicular to the robot heading direction


![image](https://github.com/user-attachments/assets/959023a1-ac94-42c2-9bfa-abc6d428a599)

![image](https://github.com/user-attachments/assets/1e7524f3-42f2-4c3a-81ce-ac8066ba61d0)

### Nonlinear Model Predictive Control
- Setting proper constraints for smoother prediction and behavior
- Terminal cost is often critical for tracking task (i.e. setting it >> stage cost)
  - but large cost might result in numerical error
