#' Sample Microclimate Dataset from the Harod Valley
#'
#' A dataset containing 7 days of hourly environmental observations and microclimate
#' predictions across three microhabitats (air at 1m, ground sun at 0cm, ground shade at 0cm)
#' in the Harod Valley, Israel.
#'
#' @format A data frame with 504 rows and 17 variables:
#' \describe{
#'   \item{sun_temp}{Observed temperature in full sun (°C)}
#'   \item{shade_temp}{Observed temperature in shade (°C)}
#'   \item{air_temp}{Observed ambient air temperature (°C)}
#'   \item{TIME}{Time in minutes from start}
#'   \item{TAREF}{Reference air temperature from meteorological forcing (°C)}
#'   \item{RH}{Relative humidity (\%)}
#'   \item{VREF}{Reference wind speed (m/s)}
#'   \item{SOLR}{Solar radiation (W/m²)}
#'   \item{TSKYC}{Sky temperature (°C)}
#'   \item{DEW}{Dew point or condensation}
#'   \item{time}{Timestamp character string}
#'   \item{time_series_doc}{Time series identifier / logger name}
#'   \item{microhabitat}{Microhabitat classification: 'air', 'shade', or 'sun'}
#'   \item{predicted}{NicheMapR physical model prediction (°C)}
#'   \item{residual}{Temperature residual (observed minus predicted, °C)}
#'   \item{Hour_sin}{Sine of diurnal cycle}
#'   \item{Hour_cos}{Cosine of diurnal cycle}
#' }
#' @source Empirical temperature loggers deployed in the Harod Valley.
#' @examples
#' data(microclimate_sample)
#' head(microclimate_sample)
"microclimate_sample"
