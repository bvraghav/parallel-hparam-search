SHELL           := /usr/bin/zsh

CONDA_ENV	:= emacs
$(info CONDA_ENV: $(CONDA_ENV))

DATA_URL	:= https://www.kaggle.com/api/v1/datasets
DATA_URL	:= $(DATA_URL)/download/prasoonkottarathil
DATA_URL	:= $(DATA_URL)/polycystic-ovary-syndrome-pcos
$(info DATA_URL: $(DATA_URL))

# To be created as per src/phs/generateHparams.py
HPARAMS_RANGE	:= 50 -3 6 7

RAW_AR		:= polycystic-ovary-syndrome-pcos.zip
AR_FILES	:= PCOS_data_without_infertility.xlsx
$(info RAW_AR: $(RAW_AR))
$(info AR_FILES: $(AR_FILES))

CONDA_ROOT	:= $(realpath $(dir $(CONDA_EXE)))
python_pre	+= source $(CONDA_ROOT)/activate $(CONDA_ENV);
python_pre	+= PYTHONPATH=src:$$PYTHONPATH
python		+= $(python_pre) python
mkdocs		+= $(python_pre) mkdocs
$(info CONDA_ROOT: $(CONDA_ROOT))
$(info python cmd: $(python))
$(info mkdocs cmd: $(mkdocs))

# $(or $(realpath dist/hparams.json),$(shell 	\
# 	$(MAKE) dist/hparams.json		\
# ))
ifneq ($(realpath dist/hparams.json),)

TRAIN_INSTANCES	:= $(shell 	\
	cat dist/hparams.json 	\
	| jq -r '.id.[]'\
)
TRAIN_INSTANCES	:= $(TRAIN_INSTANCES:%=dist/trained/%/.trained)
$(info Num TRAIN_INSTANCES: $(words $(TRAIN_INSTANCES)))
$(info Eg TRAIN_INSTANCES: $(wordlist 1,5,$(TRAIN_INSTANCES)))

else

$(warning				\
	Generate hparams using:		\
	\`make dist/hparams.json\'	\
)

endif

all : best-model

preprocess : dist/.preprocessed
train: dist/.trained
best-model : dist/.collated
generate-hparams : dist/hparams.json

dist/.preprocessed : dist/raw/$(RAW_AR) dist/preprocessed
	$(python) -m phs.cli.preprocess	\
	  --data-dir $(word 2,$(^))		\
	  -- $(<) $(AR_FILES)
	touch $(@)

dist/raw/$(RAW_AR) : dist/raw
	curl -L -o dist/raw/$(RAW_AR) $(DATA_URL)

dist dist/raw dist/preprocessed dist/trained :
	-mkdir -p $(@)

dist/hparams.json : dist
	$(python) -m phs.cli.generateHparams	\
	  -- $(@) $(HPARAMS_RANGE)

dist/.trained : $(TRAIN_INSTANCES)
	touch $(@)

dist/trained/%/.trained : dist/preprocessed dist/.preprocessed
	-mkdir -p $(dir $(@))
	$(python) -m phs.cli.train		\
	  --out-dir $(dir $(@))			\
	  --result-fname result.json		\
	  --model-fname pretrained.pkl		\
	  --hparams-path dist/hparams.json	\
	  --data-dir $(<)			\
	  --id $(*)
	touch $(@)

best-model = trained/$$(			\
  cat dist/collated_result.json 		\
  | jq -r '.best.id'				\
)
dist/.collated : dist/.trained
	$(python) -m phs.cli.collate		\
	  --out-path dist/collated_result.json	\
	  --hparams-path dist/hparams.json	\
	  --train-dir dist/trained
	unlink dist/best-model
	ln -s $(best-model) dist/best-model
	touch $(@)

### ---------------------------------------------------
### Make Documentation
### ---------------------------------------------------
localport	 = $(shell			\
  echo $$(( 1000 + ($$RANDOM % 9000) ))		\
)
PORT		:=
docserve :
	$(mkdocs) serve				\
	         -a localhost:$(or $(PORT),$(localport))

docbuild :
	$(mkdocs) build

docs : docserve
### ---------------------------------------------------
