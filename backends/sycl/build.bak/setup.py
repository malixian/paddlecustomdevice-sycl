from setuptools import setup, Distribution

packages = []
package_data = {}

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name = 'paddle-custom-sycl',
    version='0.0.1',
    description='Paddle CustomCPU plugin',
    long_description='',
    long_description_content_type="text/markdown",
    author_email="Paddle-better@baidu.com",
    maintainer="PaddlePaddle",
    maintainer_email="Paddle-better@baidu.com",
    project_urls={},
    license='Apache Software License',
    packages= [
        'paddle_custom_device',
    ],
    include_package_data=True,
    package_data = {
        '': ['*.so', '*.h', '*.py', '*.hpp'],
    },
    package_dir = {
        '': 'python',
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    entry_points={
        'console_scripts': [
        ]
    },
    classifiers=[
    ],
    keywords='Paddle CustomCPU plugin',
)
