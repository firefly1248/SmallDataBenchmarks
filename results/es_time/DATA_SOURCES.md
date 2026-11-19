# es_time: data sources

Nine datasets enter the early-stopping-on-time-data run (`early_stopping_time_plan.md`). No row of any of them is redistributed here. `scripts/es_time_prep.py` rebuilds the prepared files from the sources below and records the sha256 of every source file it reads. Licences were read from the owners' pages on 2026-11-19. "No licence found" means none was found, not that use is permitted.

| Dataset | Source | Owner and origin | Licence | Attribution |
|---|---|---|---|---|
| hotels | TidyTuesday 2020-02-11 (rfordatascience/tidytuesday) | Antonio, Almeida and Nunes, "Hotel booking demand datasets", Data in Brief 22, 2019, doi 10.1016/j.dib.2018.11.126 | CC BY 4.0 (the paper); the TidyTuesday repository is CC0 | Antonio et al. 2019 |
| himalaya | TidyTuesday 2020-09-22 | The Himalayan Database (Elizabeth Hawley and Richard Salisbury), himalayandatabase.com | Limited licence: no redistribution, prior written consent for extracts, reports or analyses | "The Himalayan Database"; consent not obtained, so no number derived from this dataset is published in this repository |
| wildlife | TidyTuesday 2019-07-23 | FAA Wildlife Strike Database, wildlife.faa.gov | No licence found | FAA Wildlife Strike Database |
| f1-dnf | TidyTuesday 2021-09-07 | Ergast Developer API, ergast.com/mrd (offline; terms read from an archive snapshot) | Non-commercial use only; database images CC BY-NC-SA 3.0 | Ergast |
| nba | fivethirtyeight/data, nba-elo/nbaallelo.csv | FiveThirtyEight; game information from Basketball-Reference (its terms were not read) | CC BY 4.0 (repository LICENSE) | FiveThirtyEight |
| elec | scikit-multiflow/streaming-datasets, elec.csv | M. Harries, "Splice-2 comparative evaluation: Electricity pricing", 1999 (New South Wales) | No licence found in the repository or for the original | Harries 1999 |
| nebraska-weather | scikit-multiflow/streaming-datasets, weather.csv | R. Elwell and R. Polikar, IEEE TNNLS 22(10), 2011 | No licence found in the repository or for the original | Elwell and Polikar 2011 |
| nycflights | vincentarelbundock/Rdatasets, nycflights13 flights | nycflights13 package (CC0 on CRAN), flight records from the Bureau of Transportation Statistics (its terms were not read) | CC0 for the CRAN package; the Rdatasets copy has no data licence | nycflights13, Bureau of Transportation Statistics |
| bank-full | UCI Bank Marketing (UCI++ in the benchmark, ARFF) | Moro, Rita and Cortez 2014, doi 10.24432/C5K306 | CC BY 4.0 | Moro, Rita and Cortez 2014, UCI Machine Learning Repository |
