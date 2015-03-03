
PYFST_PATH = fst
DEPS = src/setup.py $(PYFST_PATH)/types.yml $(PYFST_PATH)/libfst.pxd.tpl $(PYFST_PATH)/_fst.pyx.tpl $(PYFST_PATH)/_fst.pxd.tpl
all $(DEPS): 
	mustache $(PYFST_PATH)/types.yml $(PYFST_PATH)/libfst.pxd.tpl > $(PYFST_PATH)/libfst.pxd
	mustache $(PYFST_PATH)/types.yml $(PYFST_PATH)/_fst.pxd.tpl > $(PYFST_PATH)/_fst.pxd
	mustache $(PYFST_PATH)/types.yml $(PYFST_PATH)/_fst.pyx.tpl > $(PYFST_PATH)/_fst.pyx

	cython --cplus $(PYFST_PATH)/_fst.pyx
	python setup.py build

install:
	python setup.py install

clean:
	rm -rf build
	rm -f $(PYFST_PATH)/libfst.pxd
	rm -f $(PYFST_PATH)/_fst.pxd
	rm -f $(PYFST_PATH)/_fst.pyx
	rm -f $(PYFST_PATH)/libfst.cpp
	rm -f $(PYFST_PATH)/_fst.cpp
	rm -f $(PYFST_PATH)/_fst.cpp
	python setup.py clean

