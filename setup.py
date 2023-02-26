from setuptools import setup, find_packages 
  
with open('requirements.txt') as f: 
    requirements = f.readlines() 
  
long_description = '...need to add description' 
  
setup( 
        name ='stand', 
        version ='0.0.1', 
        author ='Daniel Weitekamp', 
        author_email ='weitekamp@cmu.edu', 
        url ='https://github.com/DannyWeitekamp/STAND', 
        description ='Simultaneous Tree Ambiguity Resolution and preDiction', 
        long_description = long_description, 
        long_description_content_type ="text/markdown", 
        license ='MIT', 
        packages = find_packages(), 
        classifiers =( 
            "Programming Language :: Python :: 3", 
            "License :: OSI Approved :: MIT License", 
            "Operating System :: OS Independent", 
        ), 
        keywords ='STAND numba', 
        install_requires = requirements, 
        zip_safe = False
) 
