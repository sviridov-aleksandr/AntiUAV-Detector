import os
from glob import glob
from setuptools import setup

package_name = 'uav_interceptor'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alex',
    maintainer_email='alex@example.com',
    description='ROS2 package for UAV Interceptor Visual Servoing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_node = uav_interceptor.vision_node:main',
            'video_publisher = uav_interceptor.video_publisher:main',
            'mavlink_bridge = uav_interceptor.mavlink_bridge:main',
            'wfb_video_receiver = uav_interceptor.wfb_video_receiver:main',
        ],
    },
    scripts=[],
)

# ROS2 Lyrical workaround: copy scripts to lib/<pkg>/
os.makedirs(
    os.path.join('install', package_name, 'lib', package_name),
    exist_ok=True
)