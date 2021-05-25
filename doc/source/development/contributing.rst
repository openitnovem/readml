.. _contributing:

****************************
Contributing to this package
****************************

.. contents:: Table of contents:
   :local:

.. _contributing.rules:

Some rules to keep in mind
==========================

.. _contributing.git:

Working with the code
=====================

You have an issue you want to fix, new feature to add, or documentation to improve,
you need to learn how to work with Gitlab and the package code base.

.. _contributing.version_control:

Version control, Git
--------------------

To the new user, working with Git is one of the more daunting aspects of contributing to pandas.
It can very quickly become overwhelming, but sticking to the guidelines below will help keep the process
straightforward and mostly trouble free.  As always, if you are having difficulties please
feel free to ask for help.

Forking
-------

You will need your own fork to work on the code. Go to the `package project
page <https://gitlab-repo-gpf.apps.eul.sncf.fr/digital/groupefbd-digital/90023/DSE/fbd_tools>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone https://gitlab-repo-gpf.apps.eul.sncf.fr/digital/groupefbd-digital/90023/DSE/fbd_tools.git package-yourname
    cd package-yourname
    git remote add upstream https://gitlab-repo-gpf.apps.eul.sncf.fr/digital/groupefbd-digital/90023/DSE/fbd_tools.git

This creates the directory `package-yourname` and connects your repository to
the upstream (main project) *package* repository.

Note that performing a shallow clone (with ``--depth==N``, for some ``N`` greater
or equal to 1) might break some tests and features as ``pd.show_versions()``
as the version number cannot be computed anymore.

Documentation
=============

Every documentation page must be writen in  `reStructuredText <http://sphinx-doc.org/rest.html>`_.
It can be very handy to use an online Shinx editor to write or test your `reST code <http://livesphinx.herokuapp.com/>`_

Please note that if you want to include a piece of python code that renders something like this

.. code-block:: python

   import pandas as pd
   pd.show_versions()

you must use the code-bloc directive:

::

   .. code-block:: python

      import pandas as pd
      pd.show_versions()

Remember to leave a blank line before and after the code.

Writing Howto's
---------------

bla bla bla


Docstrings
----------

Docstrings must be written with the numpy style:

::

   """
   This can be a brief description of your function

   Parameters
   ----------
   param1 : int
       First parameter; use <name of parameter> : <type of parameters>
   param2 : int
       Second parameter
   param3: str, optional
       in some cases arguments are optional
   param4: str, default None
       in case the default value None is effectively used
   
   Returns
   -------
   int
      Remember to specify the type of the variable returned
   
   Raises
   ------
   BaseException
       Describe why this error occurred

   See Also
   --------
   AFunctionName : And a comment here
   
   Examples
   --------
   >>> add(2, 2)
   4
   >>> add(25, 0)
   25
   >>> add(10, -10)
   0
   """

For a complete reference of this style, see the `link <https://numpydoc.readthedocs.io/en/latest/format.html#sections>`_


Building the doc
-----------------

