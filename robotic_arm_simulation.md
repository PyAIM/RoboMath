Of course! Let's unpack this central concept in robotics and build a comprehensive demo. I'll break it down step-by-step and provide implementations.

## **Activity Title: "Math in Motion: The Robot's Dance"**

This title captures the essence of transforming abstract mathematics into physical movement.

---

## **Part 1: Unpacking "ω = J⁻¹ v" - The Velocity Control Problem**

### **The Analogy: Driving a Car to a Destination**

Think of trying to drive from your house to a friend's house. You have:
- **Where you want to go** (the destination) → **v** (desired hand velocity)
- **How you control the car** (steering wheel, gas pedal) → **ω** (joint velocities)
- **The relationship between controls and movement** → **J** (Jacobian matrix)

### **Breaking Down the Components**

#### **1. The Robot's "Hand" Position (v)**
```python
# In our 2D world, the hand position is (x, y)
target_position = [1.5, 0.8]  # Where we want the hand to be
desired_velocity = [0.1, -0.2]  # v: How fast and in what direction we want to move
```

#### **2. The Robot's "Muscles" (ω)**
```python
# The robot moves by rotating its joints
joint_angles = [0.5, 1.2]  # θ₁, θ₂ (current shoulder and elbow angles)
joint_velocities = [0, 0]  # ω: dθ₁/dt, dθ₂/dt (how fast each joint should rotate)
```

#### **3. The Magic Translator (J) - The Jacobian**

**What is J?** It's a matrix that tells us how small changes in joint angles affect the hand position.

**Simple Example:** If I rotate my shoulder joint (θ₁) by 1 degree, how much does my hand move in the x and y directions?

**Mathematically:**
```
J = [ ∂x/∂θ₁  ∂x/∂θ₂ ]
    [ ∂y/∂θ₁  ∂y/∂θ₂ ]
```

For our 2-link arm:
```
J(θ₁, θ₂) = [ -L₁·sin(θ₁) - L₂·sin(θ₁+θ₂)   -L₂·sin(θ₁+θ₂) ]
            [  L₁·cos(θ₁) + L₂·cos(θ₁+θ₂)    L₂·cos(θ₁+θ₂) ]
```

### **The "Aha!" Moment: Why We Need J⁻¹**

We know **where we want the hand to move** (v), but our robot controller can only command **how fast to rotate the joints** (ω).

**Forward Problem (Easy):** Given joint velocities → Find hand velocity
```
v = J × ω
```

**Inverse Problem (What We Need):** Given hand velocity → Find joint velocities
```
ω = J⁻¹ × v
```

**Why is this hard?** Because J changes as the robot moves! It's like the relationship between steering and car direction changes depending on which way you're already facing.

---

## **Part 2: Detailed Implementations**

### **Python Implementation**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RobotArm2D:
    def __init__(self, L1=1.0, L2=1.0):
        self.L1 = L1  # Length of first link
        self.L2 = L2  # Length of second link
        
    def forward_kinematics(self, theta):
        """Calculate hand position from joint angles"""
        theta1, theta2 = theta
        x = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        y = self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)
        return np.array([x, y])
    
    def jacobian(self, theta):
        """Calculate the Jacobian matrix at current joint angles"""
        theta1, theta2 = theta
        
        J11 = -self.L1 * np.sin(theta1) - self.L2 * np.sin(theta1 + theta2)
        J12 = -self.L2 * np.sin(theta1 + theta2)
        J21 = self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2)
        J22 = self.L2 * np.cos(theta1 + theta2)
        
        return np.array([[J11, J12], [J21, J22]])
    
    def inverse_velocity_kinematics(self, theta, hand_velocity):
        """Calculate required joint velocities for desired hand velocity"""
        J = self.jacobian(theta)
        
        # Check if Jacobian is invertible (avoid singularities)
        if np.abs(np.linalg.det(J)) < 1e-6:
            print("Warning: Near singularity! Cannot invert Jacobian.")
            return np.array([0, 0])
        
        # ω = J⁻¹ × v
        joint_velocities = np.linalg.inv(J) @ hand_velocity
        return joint_velocities
    
    def simulate_movement(self, target_trajectory, dt=0.1):
        """Simulate the arm following a target trajectory"""
        # Initial joint angles
        theta = np.array([0.5, 1.0])
        trajectory = [theta.copy()]
        hand_positions = [self.forward_kinematics(theta)]
        
        for target_pos in target_trajectory:
            # Current hand position
            current_pos = self.forward_kinematics(theta)
            
            # Desired velocity (point toward target)
            direction = target_pos - current_pos
            if np.linalg.norm(direction) > 0.1:  # Avoid division by zero
                direction = direction / np.linalg.norm(direction)
            desired_velocity = direction * 0.5  # Constant speed
            
            # Calculate required joint velocities
            joint_vel = self.inverse_velocity_kinematics(theta, desired_velocity)
            
            # Update joint angles
            theta = theta + joint_vel * dt
            trajectory.append(theta.copy())
            hand_positions.append(self.forward_kinematics(theta))
        
        return trajectory, hand_positions

# Demo the inverse velocity control
def demo_inverse_velocity():
    arm = RobotArm2D(L1=1.0, L2=1.0)
    
    # Test at a specific configuration
    test_theta = np.array([0.5, 1.0])  # 45° shoulder, ~57° elbow
    desired_hand_velocity = np.array([0.2, 0.3])  # Move right and up
    
    print("Current joint angles:", test_theta)
    print("Desired hand velocity:", desired_hand_velocity)
    
    # Calculate required joint velocities
    joint_vel = arm.inverse_velocity_kinematics(test_theta, desired_hand_velocity)
    print("Required joint velocities:", joint_vel)
    
    # Verify: forward kinematics with these joint velocities
    J = arm.jacobian(test_theta)
    actual_hand_velocity = J @ joint_vel
    print("Actual hand velocity (should match desired):", actual_hand_velocity)
    
    # Simulate following a circular path
    t = np.linspace(0, 2*np.pi, 50)
    target_trajectory = [np.array([1.5 + 0.3*np.cos(angle), 0.5 + 0.3*np.sin(angle)]) 
                        for angle in t]
    
    trajectory, hand_positions = arm.simulate_movement(target_trajectory)
    
    # Plot results
    plot_trajectory(trajectory, hand_positions, target_trajectory)

def plot_trajectory(joint_trajectory, hand_trajectory, target_trajectory):
    """Plot the arm's movement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot arm configuration at several points
    arm = RobotArm2D()
    for i in range(0, len(joint_trajectory), 10):
        theta = joint_trajectory[i]
        pos = arm.forward_kinematics(theta)
        elbow_x = arm.L1 * np.cos(theta[0])
        elbow_y = arm.L1 * np.sin(theta[0])
        
        ax1.plot([0, elbow_x, pos[0]], [0, elbow_y, pos[1]], 'o-', alpha=0.5)
    
    # Plot hand trajectory vs target
    hand_x, hand_y = zip(*hand_trajectory)
    target_x, target_y = zip(*target_trajectory)
    
    ax1.plot(hand_x, hand_y, 'r-', label='Actual path')
    ax1.plot(target_x, target_y, 'g--', label='Target path')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Arm Motion')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot joint angles over time
    theta1, theta2 = zip(*joint_trajectory)
    time = range(len(joint_trajectory))
    
    ax2.plot(time, theta1, label='Shoulder angle (θ₁)')
    ax2.plot(time, theta2, label='Elbow angle (θ₂)')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Joint angle (radians)')
    ax2.set_title('Joint Angles Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_inverse_velocity()
```

### **MATLAB Implementation**

```matlab
classdef RobotArm2D < handle
    properties
        L1 = 1.0  % Length of first link
        L2 = 1.0  % Length of second link
    end
    
    methods
        function pos = forward_kinematics(obj, theta)
            % Calculate hand position from joint angles
            theta1 = theta(1);
            theta2 = theta(2);
            
            x = obj.L1 * cos(theta1) + obj.L2 * cos(theta1 + theta2);
            y = obj.L1 * sin(theta1) + obj.L2 * sin(theta1 + theta2);
            
            pos = [x; y];
        end
        
        function J = jacobian(obj, theta)
            % Calculate the Jacobian matrix at current joint angles
            theta1 = theta(1);
            theta2 = theta(2);
            
            J11 = -obj.L1 * sin(theta1) - obj.L2 * sin(theta1 + theta2);
            J12 = -obj.L2 * sin(theta1 + theta2);
            J21 = obj.L1 * cos(theta1) + obj.L2 * cos(theta1 + theta2);
            J22 = obj.L2 * cos(theta1 + theta2);
            
            J = [J11, J12; J21, J22];
        end
        
        function joint_vel = inverse_velocity_kinematics(obj, theta, hand_velocity)
            % Calculate required joint velocities for desired hand velocity
            J = obj.jacobian(theta);
            
            % Check if Jacobian is invertible
            if abs(det(J)) < 1e-6
                warning('Near singularity! Cannot invert Jacobian.');
                joint_vel = [0; 0];
                return;
            end
            
            % ω = J⁻¹ × v
            joint_vel = J \ hand_velocity;
        end
        
        function [trajectory, hand_positions] = simulate_movement(obj, target_trajectory, dt)
            % Simulate the arm following a target trajectory
            if nargin < 3
                dt = 0.1;
            end
            
            % Initial joint angles
            theta = [0.5; 1.0];
            trajectory = theta;
            hand_positions = obj.forward_kinematics(theta);
            
            for i = 1:size(target_trajectory, 2)
                target_pos = target_trajectory(:, i);
                
                % Current hand position
                current_pos = obj.forward_kinematics(theta);
                
                % Desired velocity (point toward target)
                direction = target_pos - current_pos;
                if norm(direction) > 0.1
                    direction = direction / norm(direction);
                end
                desired_velocity = direction * 0.5;
                
                % Calculate required joint velocities
                joint_vel = obj.inverse_velocity_kinematics(theta, desired_velocity);
                
                % Update joint angles
                theta = theta + joint_vel * dt;
                trajectory = [trajectory, theta];
                hand_positions = [hand_positions, obj.forward_kinematics(theta)];
            end
        end
    end
end

% Demo function
function demo_robot_arm()
    arm = RobotArm2D();
    
    % Test inverse velocity kinematics
    test_theta = [0.5; 1.0];
    desired_hand_velocity = [0.2; 0.3];
    
    fprintf('Current joint angles: [%.2f, %.2f]\n', test_theta);
    fprintf('Desired hand velocity: [%.2f, %.2f]\n', desired_hand_velocity);
    
    joint_vel = arm.inverse_velocity_kinematics(test_theta, desired_hand_velocity);
    fprintf('Required joint velocities: [%.2f, %.2f]\n', joint_vel);
    
    % Verify the result
    J = arm.jacobian(test_theta);
    actual_hand_velocity = J * joint_vel;
    fprintf('Actual hand velocity: [%.2f, %.2f]\n', actual_hand_velocity);
    
    % Simulate following a path
    t = linspace(0, 2*pi, 50);
    target_trajectory = [1.5 + 0.3*cos(t); 0.5 + 0.3*sin(t)];
    
    [trajectory, hand_positions] = arm.simulate_movement(target_trajectory);
    
    % Plot results
    plot_results(trajectory, hand_positions, target_trajectory, arm);
end

function plot_results(trajectory, hand_positions, target_trajectory, arm)
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot arm motion
    subplot(1, 2, 1);
    hold on;
    
    % Plot arm at several configurations
    for i = 1:5:size(trajectory, 2)
        theta = trajectory(:, i);
        pos = arm.forward_kinematics(theta);
        elbow_x = arm.L1 * cos(theta(1));
        elbow_y = arm.L1 * sin(theta(1));
        
        plot([0, elbow_x, pos(1)], [0, elbow_y, pos(2)], 'o-', 'Color', [0.7, 0.7, 0.7]);
    end
    
    % Plot trajectories
    plot(hand_positions(1, :), hand_positions(2, :), 'r-', 'LineWidth', 2, 'DisplayName', 'Actual path');
    plot(target_trajectory(1, :), target_trajectory(2, :), 'g--', 'LineWidth', 2, 'DisplayName', 'Target path');
    
    xlabel('X');
    ylabel('Y');
    title('Robot Arm Motion');
    legend;
    grid on;
    axis equal;
    
    % Plot joint angles
    subplot(1, 2, 2);
    plot(trajectory(1, :), 'b-', 'LineWidth', 2, 'DisplayName', 'Shoulder angle (θ₁)');
    hold on;
    plot(trajectory(2, :), 'r-', 'LineWidth', 2, 'DisplayName', 'Elbow angle (θ₂)');
    xlabel('Time step');
    ylabel('Joint angle (radians)');
    title('Joint Angles Over Time');
    legend;
    grid on;
end

% Run the demo
demo_robot_arm();
```

---

## **Part 3: Web Implementation with Three.js**

Yes! A web-based interactive demo would be perfect. Here's the architecture:

### **HTML Structure**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Math in Motion: The Robot's Dance</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            width: 100%;
            text-align: center;
            color: white;
            font-family: Arial, sans-serif;
        }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.8);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">Math in Motion: The Robot's Dance - Drag the target to move the arm</div>
    <div id="controls">
        <button id="resetBtn">Reset</button>
        <button id="demoBtn">Run Demo</button>
    </div>
    <script src="robot_arm.js"></script>
</body>
</html>
```

### **Three.js JavaScript Core**
```javascript
// robot_arm.js
class RobotArm3D {
    constructor(scene) {
        this.scene = scene;
        this.L1 = 2.0;  // Link lengths
        this.L2 = 2.0;
        
        // Joint angles (in radians)
        this.theta1 = Math.PI / 4;
        this.theta2 = Math.PI / 2;
        
        // Create robot arm geometry
        this.createArm();
        
        // Target sphere
        this.createTarget();
        
        // Visualization helpers
        this.createCoordinateSystem();
    }
    
    createArm() {
        // Base
        const baseGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 32);
        const baseMaterial = new THREE.MeshPhongMaterial({ color: 0x666666 });
        this.base = new THREE.Mesh(baseGeometry, baseMaterial);
        this.base.rotation.x = Math.PI / 2;
        this.scene.add(this.base);
        
        // Link 1
        const link1Geometry = new THREE.CylinderGeometry(0.1, 0.1, this.L1, 32);
        const link1Material = new THREE.MeshPhongMaterial({ color: 0x2194ce });
        this.link1 = new THREE.Mesh(link1Geometry, link1Material);
        this.link1.rotation.x = Math.PI / 2;
        this.link1.position.y = this.L1 / 2;
        this.base.add(this.link1);
        
        // Joint 1 (elbow)
        const joint1Geometry = new THREE.SphereGeometry(0.2, 32, 32);
        const joint1Material = new THREE.MeshPhongMaterial({ color: 0xff4444 });
        this.joint1 = new THREE.Mesh(joint1Geometry, joint1Material);
        this.joint1.position.y = this.L1;
        this.base.add(this.joint1);
        
        // Link 2
        const link2Geometry = new THREE.CylinderGeometry(0.08, 0.08, this.L2, 32);
        const link2Material = new THREE.MeshPhongMaterial({ color: 0x4caf50 });
        this.link2 = new THREE.Mesh(link2Geometry, link2Material);
        this.link2.rotation.x = Math.PI / 2;
        this.link2.position.y = this.L2 / 2;
        this.joint1.add(this.link2);
        
        // End effector
        const endEffectorGeometry = new THREE.SphereGeometry(0.15, 32, 32);
        const endEffectorMaterial = new THREE.MeshPhongMaterial({ color: 0xffeb3b });
        this.endEffector = new THREE.Mesh(endEffectorGeometry, endEffectorMaterial);
        this.endEffector.position.y = this.L2;
        this.joint1.add(this.endEffector);
    }
    
    createTarget() {
        const targetGeometry = new THREE.SphereGeometry(0.2, 32, 32);
        const targetMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xff00ff,
            transparent: true,
            opacity: 0.7
        });
        this.target = new THREE.Mesh(targetGeometry, targetMaterial);
        this.target.position.set(2, 2, 0);
        this.scene.add(this.target);
        
        // Add drag controls (simplified)
        this.isDragging = false;
        this.setupDragControls();
    }
    
    setupDragControls() {
        document.addEventListener('mousedown', (event) => {
            // Simple drag implementation - in practice, use raycasting
            this.isDragging = true;
        });
        
        document.addEventListener('mousemove', (event) => {
            if (this.isDragging) {
                // Update target position based on mouse
                // This is simplified - real implementation needs proper 3D projection
                this.target.position.x += event.movementX * 0.01;
                this.target.position.y -= event.movementY * 0.01;
                this.followTarget();
            }
        });
        
        document.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
    }
    
    createCoordinateSystem() {
        // Create a simple coordinate system visualization
        const axesHelper = new THREE.AxesHelper(3);
        this.scene.add(axesHelper);
    }
    
    jacobian() {
        const theta1 = this.theta1;
        const theta2 = this.theta2;
        
        const J11 = -this.L1 * Math.sin(theta1) - this.L2 * Math.sin(theta1 + theta2);
        const J12 = -this.L2 * Math.sin(theta1 + theta2);
        const J21 = this.L1 * Math.cos(theta1) + this.L2 * Math.cos(theta1 + theta2);
        const J22 = this.L2 * Math.cos(theta1 + theta2);
        
        return [
            [J11, J12],
            [J21, J22]
        ];
    }
    
    inverseVelocityKinematics(handVelocity) {
        const J = this.jacobian();
        const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        
        if (Math.abs(det) < 1e-6) {
            console.warn("Near singularity!");
            return [0, 0];
        }
        
        // Inverse of 2x2 matrix
        const invJ = [
            [J[1][1] / det, -J[0][1] / det],
            [-J[1][0] / det, J[0][0] / det]
        ];
        
        const omega1 = invJ[0][0] * handVelocity[0] + invJ[0][1] * handVelocity[1];
        const omega2 = invJ[1][0] * handVelocity[0] + invJ[1][1] * handVelocity[1];
        
        return [omega1, omega2];
    }
    
    forwardKinematics() {
        const x = this.L1 * Math.cos(this.theta1) + this.L2 * Math.cos(this.theta1 + this.theta2);
        const y = this.L1 * Math.sin(this.theta1) + this.L2 * Math.sin(this.theta1 + this.theta2);
        return [x, y];
    }
    
    followTarget() {
        const currentPos = this.forwardKinematics();
        const targetPos = [this.target.position.x, this.target.position.y];
        
        // Calculate desired velocity toward target
        const direction = [
            targetPos[0] - currentPos[0],
            targetPos[1] - currentPos[1]
        ];
        
        const distance = Math.sqrt(direction[0]**2 + direction[1]**2);
        
        if (distance > 0.1) {
            const normalizedDir = [direction[0] / distance, direction[1] / distance];
            const desiredVelocity = [normalizedDir[0] * 0.5, normalizedDir[1] * 0.5];
            
            // Calculate required joint velocities
            const jointVel = this.inverseVelocityKinematics(desiredVelocity);
            
            // Update joint angles
            this.theta1 += jointVel[0] * 0.1;
            this.theta2 += jointVel[1] * 0.1;
            
            this.updateVisualization();
        }
    }
    
    updateVisualization() {
        // Update base rotation (theta1)
        this.base.rotation.z = -this.theta1;
        
        // Update elbow joint rotation (theta2)
        this.joint1.rotation.z = -this.theta2;
        
        // Update info display
        const endPos = this.forwardKinematics();
        document.getElementById('info').innerHTML = 
            `Math in Motion: θ₁=${this.theta1.toFixed(2)}, θ₂=${this.theta2.toFixed(2)}<br>
             Hand Position: (${endPos[0].toFixed(2)}, ${endPos[1].toFixed(2)})`;
    }
    
    runDemo() {
        // Animate target in a circle
        let time = 0;
        const demoInterval = setInterval(() => {
            time += 0.1;
            this.target.position.x = 2 + Math.cos(time) * 1.5;
            this.target.position.y = 2 + Math.sin(time) * 1.5;
            this.followTarget();
            
            if (time > 2 * Math.PI) {
                clearInterval(demoInterval);
            }
        }, 100);
    }
}

// Main Three.js setup
let scene, camera, renderer, robotArm;

function init() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 8);
    camera.lookAt(0, 0, 0);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7);
    scene.add(directionalLight);
    
    // Create robot arm
    robotArm = new RobotArm3D(scene);
    robotArm.updateVisualization();
    
    // Setup controls
    document.getElementById('resetBtn').addEventListener('click', () => {
        robotArm.theta1 = Math.PI / 4;
        robotArm.theta2 = Math.PI / 2;
        robotArm.target.position.set(2, 2, 0);
        robotArm.updateVisualization();
    });
    
    document.getElementById('demoBtn').addEventListener('click', () => {
        robotArm.runDemo();
    });
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
    
    // Start animation loop
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// Start the application
init();
```

## **Key Educational Points for the Web Demo**

1. **Interactive Jacobian Visualization**: Show the Jacobian matrix updating in real-time as the arm moves
2. **Singularity Demonstration**: Let users experience configurations where J becomes non-invertible
3. **Velocity Vector Display**: Show both desired hand velocity (v) and actual achieved velocity
4. **Multiple Control Modes**: Allow switching between direct joint control and inverse velocity control

This comprehensive approach gives students an intuitive understanding of how differential equations and linear transformations enable robotic motion control, with hands-on interactive examples in multiple programming environments.