# Raw Dataset file structure
The code of this repository requires input datasets to run.
The datasets should be placed in this folder with a specific structure.
This structure is outlined below.

`kedro download` can be used to download most datasets in this folder, provided the
user has access rights and for the ones that require them credentials.

Running `tree` on this directory produces the following result:
```
.
├── adult
│   ├── adult.data
│   ├── adult.names
│   ├── adult.test
│   ├── Index
│   ├── index.html
│   └── old.adult.names
├── eicu_2_0
│   ├── admissionDrug.csv.gz
│   ├── admissionDx.csv.gz
│   ├── allergy.csv.gz
│   ├── apacheApsVar.csv.gz
│   ├── apachePatientResult.csv.gz
│   ├── apachePredVar.csv.gz
│   ├── carePlanCareProvider.csv.gz
│   ├── carePlanEOL.csv.gz
│   ├── carePlanGeneral.csv.gz
│   ├── carePlanGoal.csv.gz
│   ├── carePlanInfectiousDisease.csv.gz
│   ├── customLab.csv.gz
│   ├── diagnosis.csv.gz
│   ├── hospital.csv.gz
│   ├── index.html
│   ├── infusionDrug.csv.gz
│   ├── intakeOutput.csv.gz
│   ├── lab.csv.gz
│   ├── LICENSE.txt
│   ├── medication.csv.gz
│   ├── microLab.csv.gz
│   ├── note.csv.gz
│   ├── nurseAssessment.csv.gz
│   ├── nurseCare.csv.gz
│   ├── nurseCharting.csv.gz
│   ├── pastHistory.csv.gz
│   ├── patient.csv.gz
│   ├── physicalExam.csv.gz
│   ├── respiratoryCare.csv.gz
│   ├── respiratoryCharting.csv.gz
│   ├── SHA256SUMS.txt
│   ├── treatment.csv.gz
│   ├── vitalAperiodic.csv.gz
│   └── vitalPeriodic.csv.gz
├── mimiciv_1_0
│   ├── core
│   │   ├── admissions.csv.gz
│   │   ├── index.html
│   │   ├── patients.csv.gz
│   │   └── transfers.csv.gz
│   ├── hosp
│   │   ├── d_hcpcs.csv.gz
│   │   ├── diagnoses_icd.csv.gz
│   │   ├── d_icd_diagnoses.csv.gz
│   │   ├── d_icd_procedures.csv.gz
│   │   ├── d_labitems.csv.gz
│   │   ├── drgcodes.csv.gz
│   │   ├── emar.csv.gz
│   │   ├── emar_detail.csv.gz
│   │   ├── hcpcsevents.csv.gz
│   │   ├── index.html
│   │   ├── labevents.csv.gz
│   │   ├── microbiologyevents.csv.gz
│   │   ├── pharmacy.csv.gz
│   │   ├── poe.csv.gz
│   │   ├── poe_detail.csv.gz
│   │   ├── prescriptions.csv.gz
│   │   ├── procedures_icd.csv.gz
│   │   └── services.csv.gz
│   ├── icu
│   │   ├── chartevents.csv
│   │   ├── datetimeevents.csv.gz
│   │   ├── d_items.csv.gz
│   │   ├── icustays.csv.gz
│   │   ├── index.html
│   │   ├── inputevents.csv.gz
│   │   ├── outputevents.csv.gz
│   │   └── procedureevents.csv.gz
│   ├── index.html
│   ├── LICENSE.txt
│   └── SHA256SUMS.txt
├── mimiciv_ed_1_0
│   ├── ed
│   │   ├── diagnosis.csv.gz
│   │   ├── edstays.csv.gz
│   │   ├── index.html
│   │   ├── medrecon.csv.gz
│   │   ├── pyxis.csv.gz
│   │   ├── triage.csv.gz
│   │   └── vitalsign.csv.gz
│   ├── index.html
│   ├── LICENSE.txt
│   ├── README.txt
│   └── SHA256SUMS.txt
├── readme.md
└── texas
    ├── Facility_type1q2011_tab.zip
    ├── Facility_type1q2012_tab.zip
    ├── Facility_type1q2013_tab.zip
    ├── Facility_type1q2014_tab.zip
    ├── Facility_type1q2015_tab.zip
    ├── Facility_type2q2011_tab.zip
    ├── Facility_type2q2012_tab.zip
    ├── Facility_type2q2013_tab.zip
    ├── Facility_type2q2014_tab.zip
    ├── Facility_type2q2015_tab.zip
    ├── Facility_type3q2011_tab.zip
    ├── Facility_type3q2012_tab.zip
    ├── Facility_type3q2013_tab.zip
    ├── Facility_type3q2014_tab.zip
    ├── Facility_type3q2015_tab.zip
    ├── Facility_type4q2011_tab.zip
    ├── Facility_type4q2012_tab.zip
    ├── Facility_type4q2013_tab.zip
    ├── Facility_type4q2014_tab.zip
    ├── Facility_type4q2015_tab.zip
    ├── PUDF-1Q2006-tab-delimited.zip
    ├── PUDF-1Q2007-tab-delimited.zip
    ├── PUDF-1Q2008-tab-delimited.zip
    ├── PUDF-1Q2009-tab-delimited.zip
    ├── PUDF-1Q2010-tab-delimited.zip
    ├── PUDF-2Q2006-tab-delimited.zip
    ├── PUDF-2Q2007-tab-delimited.zip
    ├── PUDF-2Q2008-tab-delimited.zip
    ├── PUDF-2Q2009-tab-delimited.zip
    ├── PUDF-2Q2010-tab-delimited.zip
    ├── PUDF-3Q2006-tab-delimited.zip
    ├── PUDF-3Q2007-tab-delimited.zip
    ├── PUDF-3Q2008-tab-delimited.zip
    ├── PUDF-3Q2009-tab-delimited.zip
    ├── PUDF-3Q2010-tab-delimited.zip
    ├── PUDF-4Q2006-tab-delimited.zip
    ├── PUDF-4Q2007-tab-delimited.zip
    ├── PUDF-4Q2008-tab-delimited.zip
    ├── PUDF-4Q2009-tab-delimited.zip
    ├── PUDF-4Q2010-tab-delimited.zip
    ├── PUDF_base1_1q2011_tab.zip
    ├── PUDF_base1_1q2012_tab.zip
    ├── PUDF_base1_1q2013_tab.zip
    ├── PUDF_base1_1q2014_tab.zip
    ├── PUDF_base1_1q2015_tab.zip
    ├── PUDF_base1_2q2011_tab.zip
    ├── PUDF_base1_2q2012_tab.zip
    ├── PUDF_base1_2q2013_tab.zip
    ├── PUDF_base1_2q2014_tab.zip
    ├── PUDF_base1_2q2015_tab.zip
    ├── PUDF_base1_3q2011_tab.zip
    ├── PUDF_base1_3q2012_tab.zip
    ├── PUDF_base1_3q2013_tab.zip
    ├── PUDF_base1_3q2014_tab.zip
    ├── PUDF_base1_3q2015_tab.zip
    ├── PUDF_base1_4q2011_tab.zip
    ├── PUDF_base1_4q2012_tab.zip
    ├── PUDF_base1_4q2013_tab.zip
    ├── PUDF_base1_4q2014_tab.zip
    ├── PUDF_base1_4q2015_tab.zip
    ├── PUDF_base2_1q2011_tab.zip
    ├── PUDF_base2_1q2012_tab.zip
    ├── PUDF_base2_1q2013_tab.zip
    ├── PUDF_base2_1q2014_tab.zip
    ├── PUDF_base2_1q2015_tab.zip
    ├── PUDF_base2_2q2011_tab.zip
    ├── PUDF_base2_2q2012_tab.zip
    ├── PUDF_base2_2q2013_tab.zip
    ├── PUDF_base2_2q2014_tab.zip
    ├── PUDF_base2_2q2015_tab.zip
    ├── PUDF_base2_3q2011_tab.zip
    ├── PUDF_base2_3q2012_tab.zip
    ├── PUDF_base2_3q2013_tab.zip
    ├── PUDF_base2_3q2014_tab.zip
    ├── PUDF_base2_3q2015_tab.zip
    ├── PUDF_base2_4q2011_tab.zip
    ├── PUDF_base2_4q2012_tab.zip
    ├── PUDF_base2_4q2013_tab.zip
    ├── PUDF_base2_4q2014_tab.zip
    ├── PUDF_base2_4q2015_tab.zip
    ├── PUDF_charges1q2011_tab.zip
    ├── PUDF_charges1q2012_tab.zip
    ├── PUDF_charges1q2013_tab.zip
    ├── PUDF_charges1q2014_tab.zip
    ├── PUDF_charges1q2015_tab.zip
    ├── PUDF_charges2q2011_tab.zip
    ├── PUDF_charges2q2012_tab.zip
    ├── PUDF_charges2q2013_tab.zip
    ├── PUDF_charges2q2014_tab.zip
    ├── PUDF_charges2q2015_tab.zip
    ├── PUDF_charges3q2011_tab.zip
    ├── PUDF_charges3q2012_tab.zip
    ├── PUDF_charges3q2013_tab.zip
    ├── PUDF_charges3q2014_tab.zip
    ├── PUDF_charges3q2015_tab.zip
    ├── PUDF_charges4q2011_tab.zip
    ├── PUDF_charges4q2012_tab.zip
    ├── PUDF_charges4q2013_tab.zip
    ├── PUDF_charges4q2014_tab.zip
    └── PUDF_charges4q2015_tab.zip

9 directories, 185 files
```