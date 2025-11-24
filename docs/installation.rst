Installation
============

Scikit-Topt is a Python library distributed on PyPI as the package
``scikit-topt`` (imported as ``sktopt``).  
It supports **Python 3.10–3.13**:

- **Python 3.10–3.12**: fully supported and tested.
- **Python 3.13**: core topology optimization works as normal,  
  but VTK-based features (VTU export and image rendering) require
  PyVista/VTK, which do not currently provide wheels for Python 3.13.

**Install sktopt**:  
Open your terminal or command prompt and run one of the following commands:

- Using pip::

      pip install scikit-topt

- Using poetry::

      poetry add scikit-topt

**Verify the installation**:  
After the installation is complete, you can verify that sktopt is installed correctly by running the following command::

    python -c "import sktopt; print(sktopt.__version__)"
