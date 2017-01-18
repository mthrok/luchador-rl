#!/bin/bash

echo "THEANO_FLAGS='allow_gc=False,floatX=float32' luchador exercise debug/RPiRoverEnv.yml --agent debug/RPiRoverAgent.yml --step 100 --episode 1000 --report 50"
