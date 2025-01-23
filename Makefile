## Manage python dependencies:
##
## make help   -- Display this message
## make all    -- Recompute the .txt requirements files, keeping the
##                pinned package versions.  Use this after adding or
##                removing packages from the .in files.
## make update -- Recompute the .txt requirements files files from
##                scratch, updating all packages unless pinned in the
##                .in files.


DEVELOPER_ENV := requirements-dev.in

PIP_COMPILE := pip-compile -q --no-header --resolver=backtracking
CONSTRAINTS_ENV := $(addsuffix .txt, $(basename $(DEVELOPER_ENV)))
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
SOURCES := $(shell find . -name 'requirements*.in' -not -path '*/cdk.out/*' -not -name $(DEVELOPER_ENV))

help:
	@sed -rn 's/^## ?//;T;p' $(MAKEFILE_LIST)

$(CONSTRAINTS_ENV): $(SOURCES)
	CONSTRAINTS=/dev/null $(PIP_COMPILE) --strip-extras -o $@ $^ $(DEVELOPER_ENV)

%.txt: %.in 
	CONSTRAINTS=$(CURRENT_DIR)/$(CONSTRAINTS_ENV) $(PIP_COMPILE) --no-strip-extras --no-annotate -o $@ $<

all: $(CONSTRAINTS_ENV) $(addsuffix .txt, $(basename $(SOURCES)))

clean:
	rm -rf $(CONSTRAINTS_ENV) $(addsuffix .txt, $(basename $(SOURCES)))

protoc:
	python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./physguide/vesseldata/vesseldata.proto --mypy_out=.
	python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./physguide/forecast/forecast.proto --mypy_out=.

update: clean all

.PHONY: help all clean update
