from setuptools import setup

package_name = 'ad_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vivek Alumootil',
    maintainer_email='vivekalumootil@gmail.com',
    description='Generates movement controls based on webcam stream',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc = ad_mpc.mpc:main',
            'mpc_hough = ad_mpc.mpc_hough:main',
            'mpc_true = ad_mpc.mpc_true:main'
        ],
    },
)
