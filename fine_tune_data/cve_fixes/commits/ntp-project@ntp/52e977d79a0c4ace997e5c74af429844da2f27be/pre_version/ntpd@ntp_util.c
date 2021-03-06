/*
 * ntp_util.c - stuff I didn't have any other place for
 */
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "ntpd.h"
#include "ntp_unixtime.h"
#include "ntp_filegen.h"
#include "ntp_if.h"
#include "ntp_stdlib.h"
#include "ntp_assert.h"
#include "ntp_calendar.h"
#include "lib_strbuf.h"

#include <stdio.h>
#include <ctype.h>
#include <sys/types.h>
#ifdef HAVE_SYS_IOCTL_H
# include <sys/ioctl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#ifdef HAVE_IEEEFP_H
# include <ieeefp.h>
#endif
#ifdef HAVE_MATH_H
# include <math.h>
#endif

#ifdef	DOSYNCTODR
# if !defined(VMS)
#  include <sys/resource.h>
# endif /* VMS */
#endif

#if defined(VMS)
# include <descrip.h>
#endif /* VMS */

/*
 * Defines used by the leapseconds stuff
 */
#define	MAX_TAI	100			/* max TAI offset (s) */
#define	L_DAY	86400UL			/* seconds per day */
#define	L_YEAR	(L_DAY * 365)		/* days per year */
#define	L_LYEAR	(L_YEAR + L_DAY)	/* days per leap year */
#define	L_4YEAR	(L_LYEAR + 3 * L_YEAR)	/* days per leap cycle */
#define	L_CENT	(L_4YEAR * 25)		/* days per century */

/*
 * This contains odds and ends, including the hourly stats, various
 * configuration items, leapseconds stuff, etc.
 */
/*
 * File names
 */
static	char *key_file_name;		/* keys file name */
char	*leapseconds_file_name;		/* leapseconds file name */
char	*stats_drift_file;		/* frequency file name */
static	char *stats_temp_file;		/* temp frequency file name */
static double wander_resid;		/* last frequency update */
double	wander_threshold = 1e-7;	/* initial frequency threshold */

/*
 * Statistics file stuff
 */
#ifndef NTP_VAR
# ifndef SYS_WINNT
#  define NTP_VAR "/var/NTP/"		/* NOTE the trailing '/' */
# else
#  define NTP_VAR "c:\\var\\ntp\\"	/* NOTE the trailing '\\' */
# endif /* SYS_WINNT */
#endif

#ifndef MAXPATHLEN
# define MAXPATHLEN 256
#endif

#ifdef DEBUG_TIMING
static FILEGEN timingstats;
#endif
#ifdef AUTOKEY
static FILEGEN cryptostats;
#endif	/* AUTOKEY */

static	char statsdir[MAXPATHLEN] = NTP_VAR;
static FILEGEN peerstats;
static FILEGEN loopstats;
static FILEGEN clockstats;
static FILEGEN rawstats;
static FILEGEN sysstats;
static FILEGEN protostats;

/*
 * This controls whether stats are written to the fileset. Provided
 * so that ntpdc can turn off stats when the file system fills up. 
 */
int stats_control;

/*
 * Last frequency written to file.
 */
static double prev_drift_comp;		/* last frequency update */

/*
 * Function prototypes
 */
static	int	leap_file(FILE *);
static	void	record_sys_stats(void);
	void	ntpd_time_stepped(void);

/* 
 * Prototypes
 */
#ifdef DEBUG
void	uninit_util(void);
#endif


/*
 * uninit_util - free memory allocated by init_util
 */
#ifdef DEBUG
void
uninit_util(void)
{
#if defined(_MSC_VER) && defined (_DEBUG)
	_CrtCheckMemory();
#endif
	if (stats_drift_file) {
		free(stats_drift_file);
		free(stats_temp_file);
		stats_drift_file = NULL;
		stats_temp_file = NULL;
	}
	if (key_file_name) {
		free(key_file_name);
		key_file_name = NULL;
	}
	filegen_unregister("peerstats");
	filegen_unregister("loopstats");
	filegen_unregister("clockstats");
	filegen_unregister("rawstats");
	filegen_unregister("sysstats");
	filegen_unregister("protostats");
#ifdef AUTOKEY
	filegen_unregister("cryptostats");
#endif	/* AUTOKEY */
#ifdef DEBUG_TIMING
	filegen_unregister("timingstats");
#endif	/* DEBUG_TIMING */

#if defined(_MSC_VER) && defined (_DEBUG)
	_CrtCheckMemory();
#endif
}
#endif /* DEBUG */


/*
 * init_util - initialize the util module of ntpd
 */
void
init_util(void)
{
	filegen_register(statsdir, "peerstats",	  &peerstats);
	filegen_register(statsdir, "loopstats",	  &loopstats);
	filegen_register(statsdir, "clockstats",  &clockstats);
	filegen_register(statsdir, "rawstats",	  &rawstats);
	filegen_register(statsdir, "sysstats",	  &sysstats);
	filegen_register(statsdir, "protostats",  &protostats);
#ifdef AUTOKEY
	filegen_register(statsdir, "cryptostats", &cryptostats);
#endif	/* AUTOKEY */
#ifdef DEBUG_TIMING
	filegen_register(statsdir, "timingstats", &timingstats);
#endif	/* DEBUG_TIMING */
	/*
	 * register with libntp ntp_set_tod() to call us back
	 * when time is stepped.
	 */
	step_callback = &ntpd_time_stepped;
#ifdef DEBUG
	atexit(&uninit_util);
#endif /* DEBUG */
}


/*
 * hourly_stats - print some interesting stats
 */
void
write_stats(void)
{
	FILE	*fp;
#ifdef DOSYNCTODR
	struct timeval tv;
#if !defined(VMS)
	int	prio_set;
#endif
#ifdef HAVE_GETCLOCK
	struct timespec ts;
#endif
	int	o_prio;

	/*
	 * Sometimes having a Sun can be a drag.
	 *
	 * The kernel variable dosynctodr controls whether the system's
	 * soft clock is kept in sync with the battery clock. If it
	 * is zero, then the soft clock is not synced, and the battery
	 * clock is simply left to rot. That means that when the system
	 * reboots, the battery clock (which has probably gone wacky)
	 * sets the soft clock. That means ntpd starts off with a very
	 * confused idea of what time it is. It then takes a large
	 * amount of time to figure out just how wacky the battery clock
	 * has made things drift, etc, etc. The solution is to make the
	 * battery clock sync up to system time. The way to do THAT is
	 * to simply set the time of day to the current time of day, but
	 * as quickly as possible. This may, or may not be a sensible
	 * thing to do.
	 *
	 * CAVEAT: settimeofday() steps the sun clock by about 800 us,
	 *	   so setting DOSYNCTODR seems a bad idea in the
	 *	   case of us resolution
	 */

#if !defined(VMS)
	/*
	 * (prr) getpriority returns -1 on error, but -1 is also a valid
	 * return value (!), so instead we have to zero errno before the
	 * call and check it for non-zero afterwards.
	 */
	errno = 0;
	prio_set = 0;
	o_prio = getpriority(PRIO_PROCESS,0); /* Save setting */

	/*
	 * (prr) if getpriority succeeded, call setpriority to raise
	 * scheduling priority as high as possible.  If that succeeds
	 * as well, set the prio_set flag so we remember to reset
	 * priority to its previous value below.  Note that on Solaris
	 * 2.6 (and beyond?), both getpriority and setpriority will fail
	 * with ESRCH, because sched_setscheduler (called from main) put
	 * us in the real-time scheduling class which setpriority
	 * doesn't know about. Being in the real-time class is better
	 * than anything setpriority can do, anyhow, so this error is
	 * silently ignored.
	 */
	if ((errno == 0) && (setpriority(PRIO_PROCESS,0,-20) == 0))
		prio_set = 1;	/* overdrive */
#endif /* VMS */
#ifdef HAVE_GETCLOCK
	(void) getclock(TIMEOFDAY, &ts);
	tv.tv_sec = ts.tv_sec;
	tv.tv_usec = ts.tv_nsec / 1000;
#else /*  not HAVE_GETCLOCK */
	GETTIMEOFDAY(&tv,(struct timezone *)NULL);
#endif /* not HAVE_GETCLOCK */
	if (ntp_set_tod(&tv,(struct timezone *)NULL) != 0)
		msyslog(LOG_ERR, "can't sync battery time: %m");
#if !defined(VMS)
	if (prio_set)
		setpriority(PRIO_PROCESS, 0, o_prio); /* downshift */
#endif /* VMS */
#endif /* DOSYNCTODR */
	record_sys_stats();
	if (stats_drift_file != 0) {

		/*
		 * When the frequency file is written, initialize the
		 * prev_drift_comp and wander_resid. Thereafter,
		 * reduce the wander_resid by half each hour. When
		 * the difference between the prev_drift_comp and
		 * drift_comp is less than the wander_resid, update
		 * the frequncy file. This minimizes the file writes to
		 * nonvolaile storage.
		 */
#ifdef DEBUG
		if (debug)
			printf("write_stats: frequency %.6lf thresh %.6lf, freq %.6lf\n",
			    (prev_drift_comp - drift_comp) * 1e6, wander_resid *
			    1e6, drift_comp * 1e6);
#endif
		if (fabs(prev_drift_comp - drift_comp) < wander_resid) {
			wander_resid *= 0.5;
			return;
		}
		prev_drift_comp = drift_comp;
		wander_resid = wander_threshold;
		if ((fp = fopen(stats_temp_file, "w")) == NULL) {
			msyslog(LOG_ERR, "frequency file %s: %m",
			    stats_temp_file);
			return;
		}
		fprintf(fp, "%.3f\n", drift_comp * 1e6);
		(void)fclose(fp);
		/* atomic */
#ifdef SYS_WINNT
		if (_unlink(stats_drift_file)) /* rename semantics differ under NT */
			msyslog(LOG_WARNING, 
				"Unable to remove prior drift file %s, %m", 
				stats_drift_file);
#endif /* SYS_WINNT */

#ifndef NO_RENAME
		if (rename(stats_temp_file, stats_drift_file))
			msyslog(LOG_WARNING, 
				"Unable to rename temp drift file %s to %s, %m", 
				stats_temp_file, stats_drift_file);
#else
		/* we have no rename NFS of ftp in use */
		if ((fp = fopen(stats_drift_file, "w")) ==
		    NULL) {
			msyslog(LOG_ERR,
			    "frequency file %s: %m",
			    stats_drift_file);
			return;
		}
#endif

#if defined(VMS)
		/* PURGE */
		{
			$DESCRIPTOR(oldvers,";-1");
			struct dsc$descriptor driftdsc = {
				strlen(stats_drift_file), 0, 0,
				    stats_drift_file };
			while(lib$delete_file(&oldvers,
			    &driftdsc) & 1);
		}
#endif
	}
}


/*
 * stats_config - configure the stats operation
 */
void
stats_config(
	int item,
	const char *invalue	/* only one type so far */
	)
{
	FILE	*fp;
	const char *value;
	int	len;
	double	old_drift;
#ifndef VMS
	const char temp_ext[] = ".TEMP";
#else
	const char temp_ext[] = "-TEMP";
#endif

	/*
	 * Expand environment strings under Windows NT, since the
	 * command interpreter doesn't do this, the program must.
	 */
#ifdef SYS_WINNT
	char newvalue[MAX_PATH], parameter[MAX_PATH];

	if (!ExpandEnvironmentStrings(invalue, newvalue, MAX_PATH)) {
		switch (item) {
		case STATS_FREQ_FILE:
			strncpy(parameter, "STATS_FREQ_FILE",
				sizeof(parameter));
			break;

		case STATS_LEAP_FILE:
			strncpy(parameter, "STATS_LEAP_FILE",
				sizeof(parameter));
			break;

		case STATS_STATSDIR:
			strncpy(parameter, "STATS_STATSDIR",
				sizeof(parameter));
			break;

		case STATS_PID_FILE:
			strncpy(parameter, "STATS_PID_FILE",
				sizeof(parameter));
			break;

		default:
			strncpy(parameter, "UNKNOWN",
				sizeof(parameter));
			break;
		}
		value = invalue;
		msyslog(LOG_ERR,
			"ExpandEnvironmentStrings(%s) failed: %m\n",
			parameter);
	} else {
		value = newvalue;
	}
#else	 
	value = invalue;
#endif /* SYS_WINNT */

	switch (item) {

	/*
	 * Open and read frequency file.
	 */
	case STATS_FREQ_FILE:
		if (!value || (len = strlen(value)) == 0)
			break;

		stats_drift_file = erealloc(stats_drift_file, len + 1);
		stats_temp_file = erealloc(stats_temp_file, 
		    len + sizeof(".TEMP"));
		memcpy(stats_drift_file, value, (size_t)(len+1));
		memcpy(stats_temp_file, value, (size_t)len);
		memcpy(stats_temp_file + len, temp_ext, sizeof(temp_ext));

		/*
		 * Open drift file and read frequency. If the file is
		 * missing or contains errors, tell the loop to reset.
		 */
		if ((fp = fopen(stats_drift_file, "r")) == NULL)
			break;

		if (fscanf(fp, "%lf", &old_drift) != 1) {
			msyslog(LOG_ERR,
				"format error frequency file %s", 
				stats_drift_file);
			fclose(fp);
			break;

		}
		fclose(fp);
		loop_config(LOOP_FREQ, old_drift);
		prev_drift_comp = drift_comp;
		break;

	/*
	 * Specify statistics directory.
	 */
	case STATS_STATSDIR:

		/*
		 * HMS: the following test is insufficient:
		 * - value may be missing the DIR_SEP
		 * - we still need the filename after it
		 */
		if (strlen(value) >= sizeof(statsdir)) {
			msyslog(LOG_ERR,
			    "statsdir too long (>%d, sigh)",
			    (int)sizeof(statsdir) - 1);
		} else {
			l_fp now;
			int add_dir_sep;
			int value_l = strlen(value);

			/* Add a DIR_SEP unless we already have one. */
			if (value_l == 0)
				add_dir_sep = 0;
			else
				add_dir_sep = (DIR_SEP !=
				    value[value_l - 1]);

			if (add_dir_sep)
				snprintf(statsdir, sizeof(statsdir),
				    "%s%c", value, DIR_SEP);
			else
				snprintf(statsdir, sizeof(statsdir),
				    "%s", value);
			get_systime(&now);
			if (peerstats.prefix == &statsdir[0] &&
			    peerstats.fp != NULL) {
				fclose(peerstats.fp);
				peerstats.fp = NULL;
				filegen_setup(&peerstats, now.l_ui);
			}
			if (loopstats.prefix == &statsdir[0] &&
			    loopstats.fp != NULL) {
				fclose(loopstats.fp);
				loopstats.fp = NULL;
				filegen_setup(&loopstats, now.l_ui);
			}
			if (clockstats.prefix == &statsdir[0] &&
			    clockstats.fp != NULL) {
				fclose(clockstats.fp);
				clockstats.fp = NULL;
				filegen_setup(&clockstats, now.l_ui);
			}
			if (rawstats.prefix == &statsdir[0] &&
			    rawstats.fp != NULL) {
				fclose(rawstats.fp);
				rawstats.fp = NULL;
				filegen_setup(&rawstats, now.l_ui);
			}
			if (sysstats.prefix == &statsdir[0] &&
			    sysstats.fp != NULL) {
				fclose(sysstats.fp);
				sysstats.fp = NULL;
				filegen_setup(&sysstats, now.l_ui);
			}
			if (protostats.prefix == &statsdir[0] &&
			    protostats.fp != NULL) {
				fclose(protostats.fp);
				protostats.fp = NULL;
				filegen_setup(&protostats, now.l_ui);
			}
#ifdef AUTOKEY
			if (cryptostats.prefix == &statsdir[0] &&
			    cryptostats.fp != NULL) {
				fclose(cryptostats.fp);
				cryptostats.fp = NULL;
				filegen_setup(&cryptostats, now.l_ui);
			}
#endif	/* AUTOKEY */
#ifdef DEBUG_TIMING
			if (timingstats.prefix == &statsdir[0] &&
			    timingstats.fp != NULL) {
				fclose(timingstats.fp);
				timingstats.fp = NULL;
				filegen_setup(&timingstats, now.l_ui);
			}
#endif	/* DEBUG_TIMING */
		}
		break;

	/*
	 * Open pid file.
	 */
	case STATS_PID_FILE:
		if ((fp = fopen(value, "w")) == NULL) {
			msyslog(LOG_ERR, "pid file %s: %m",
			    value);
			break;
		}
		fprintf(fp, "%d", (int)getpid());
		fclose(fp);;
		break;

	/*
	 * Read leapseconds file.
	 */
	case STATS_LEAP_FILE:
		if ((fp = fopen(value, "r")) == NULL) {
			msyslog(LOG_ERR, "leapseconds file %s: %m",
			    value);
			break;
		}

		if (leap_file(fp) < 0)
			msyslog(LOG_ERR,
			    "format error leapseconds file %s",
			    value);
		else
			mprintf_event(EVNT_TAI, NULL,
				      "%d leap %s expire %s", leap_tai,
				      fstostr(leap_sec),
				      fstostr(leap_expire));
		fclose(fp);
		break;

	default:
		/* oh well */
		break;
	}
}


/*
 * record_peer_stats - write peer statistics to file
 *
 * file format:
 * day (MJD)
 * time (s past UTC midnight)
 * IP address
 * status word (hex)
 * offset
 * delay
 * dispersion
 * jitter
*/
void
record_peer_stats(
	sockaddr_u *addr,
	int	status,
	double	offset,		/* offset */
	double	delay,		/* delay */
	double	dispersion,	/* dispersion */
	double	jitter		/* jitter */
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&peerstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (peerstats.fp != NULL) {
		fprintf(peerstats.fp,
		    "%lu %s %s %x %.9f %.9f %.9f %.9f\n", day,
		    ulfptoa(&now, 3), stoa(addr), status, offset,
		    delay, dispersion, jitter);
		fflush(peerstats.fp);
	}
}


/*
 * record_loop_stats - write loop filter statistics to file
 *
 * file format:
 * day (MJD)
 * time (s past midnight)
 * offset
 * frequency (PPM)
 * jitter
 * wnder (PPM)
 * time constant (log2)
 */
void
record_loop_stats(
	double	offset,		/* offset */
	double	freq,		/* frequency (PPM) */
	double	jitter,		/* jitter */
	double	wander,		/* wander (PPM) */
	int spoll
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&loopstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (loopstats.fp != NULL) {
		fprintf(loopstats.fp, "%lu %s %.9f %.3f %.9f %.6f %d\n",
		    day, ulfptoa(&now, 3), offset, freq * 1e6, jitter,
		    wander * 1e6, spoll);
		fflush(loopstats.fp);
	}
}


/*
 * record_clock_stats - write clock statistics to file
 *
 * file format:
 * day (MJD)
 * time (s past midnight)
 * IP address
 * text message
 */
void
record_clock_stats(
	sockaddr_u *addr,
	const char *text	/* timecode string */
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&clockstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (clockstats.fp != NULL) {
		fprintf(clockstats.fp, "%lu %s %s %s\n", day,
		    ulfptoa(&now, 3), stoa(addr), text);
		fflush(clockstats.fp);
	}
}


/*
 * mprintf_clock_stats - write clock statistics to file with
 *			msnprintf-style formatting.
 */
int
mprintf_clock_stats(
	sockaddr_u *addr,
	const char *fmt,
	...
	)
{
	va_list	ap;
	int	rc;
	char	msg[512];

	va_start(ap, fmt);
	rc = mvsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
	if (stats_control)
		record_clock_stats(addr, msg);

	return rc;
}

/*
 * record_raw_stats - write raw timestamps to file
 *
 * file format
 * day (MJD)
 * time (s past midnight)
 * peer ip address
 * IP address
 * t1 t2 t3 t4 timestamps
 */
void
record_raw_stats(
	sockaddr_u *srcadr,
	sockaddr_u *dstadr,
	l_fp	*t1,		/* originate timestamp */
	l_fp	*t2,		/* receive timestamp */
	l_fp	*t3,		/* transmit timestamp */
	l_fp	*t4		/* destination timestamp */
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&rawstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (rawstats.fp != NULL) {
		fprintf(rawstats.fp, "%lu %s %s %s %s %s %s %s\n", day,
		    ulfptoa(&now, 3), stoa(srcadr), dstadr ? 
		    stoa(dstadr) : "-",	ulfptoa(t1, 9), ulfptoa(t2, 9),
		    ulfptoa(t3, 9), ulfptoa(t4, 9));
		fflush(rawstats.fp);
	}
}


/*
 * record_sys_stats - write system statistics to file
 *
 * file format
 * day (MJD)
 * time (s past midnight)
 * time since reset
 * packets recieved
 * packets for this host
 * current version
 * old version
 * access denied
 * bad length or format
 * bad authentication
 * declined
 * rate exceeded
 * KoD sent
 */
void
record_sys_stats(void)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&sysstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (sysstats.fp != NULL) {
		fprintf(sysstats.fp,
		    "%lu %s %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n",
		    day, ulfptoa(&now, 3), current_time - sys_stattime,
		    sys_received, sys_processed, sys_newversion,
		    sys_oldversion, sys_restricted, sys_badlength,
		    sys_badauth, sys_declined, sys_limitrejected,
		    sys_kodsent);
		fflush(sysstats.fp);
		proto_clr_stats();
	}
}


/*
 * record_proto_stats - write system statistics to file
 *
 * file format
 * day (MJD)
 * time (s past midnight)
 * text message
 */
void
record_proto_stats(
	char	*str		/* text string */
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&protostats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (protostats.fp != NULL) {
		fprintf(protostats.fp, "%lu %s %s\n", day,
		    ulfptoa(&now, 3), str);
		fflush(protostats.fp);
	}
}


#ifdef AUTOKEY
/*
 * record_crypto_stats - write crypto statistics to file
 *
 * file format:
 * day (mjd)
 * time (s past midnight)
 * peer ip address
 * text message
 */
void
record_crypto_stats(
	sockaddr_u *addr,
	const char *text	/* text message */
	)
{
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&cryptostats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (cryptostats.fp != NULL) {
		if (addr == NULL)
			fprintf(cryptostats.fp, "%lu %s 0.0.0.0 %s\n",
			    day, ulfptoa(&now, 3), text);
		else
			fprintf(cryptostats.fp, "%lu %s %s %s\n",
			    day, ulfptoa(&now, 3), stoa(addr), text);
		fflush(cryptostats.fp);
	}
}
#endif	/* AUTOKEY */


#ifdef DEBUG_TIMING
/*
 * record_timing_stats - write timing statistics to file
 *
 * file format:
 * day (mjd)
 * time (s past midnight)
 * text message
 */
void
record_timing_stats(
	const char *text	/* text message */
	)
{
	static unsigned int flshcnt;
	l_fp	now;
	u_long	day;

	if (!stats_control)
		return;

	get_systime(&now);
	filegen_setup(&timingstats, now.l_ui);
	day = now.l_ui / 86400 + MJD_1900;
	now.l_ui %= 86400;
	if (timingstats.fp != NULL) {
		fprintf(timingstats.fp, "%lu %s %s\n", day, lfptoa(&now,
		    3), text);
		if (++flshcnt % 100 == 0)
			fflush(timingstats.fp);
	}
}
#endif


/*
 * leap_file - read leapseconds file
 *
 * Read the ERTS leapsecond file in NIST text format and extract the
 * NTP seconds of the latest leap and TAI offset after the leap.
 */
static int
leap_file(
	FILE	*fp		/* file handle */
	)
{
	char	buf[NTP_MAXSTRLEN]; /* file line buffer */
	u_long	leap;		/* NTP time at leap */
	u_long	expire;		/* NTP time when file expires */
	int	offset;		/* TAI offset at leap (s) */
	int	i;

	/*
	 * Read and parse the leapseconds file. Empty lines and comments
	 * are ignored. A line beginning with #@ contains the file
	 * expiration time in NTP seconds. Other lines begin with two
	 * integers followed by junk or comments. The first integer is
	 * the NTP seconds at the leap, the second is the TAI offset
	 * after the leap.
	 */
	offset = 0;
	leap = 0;
	expire = 0;
	i = 10;
	while (fgets(buf, NTP_MAXSTRLEN - 1, fp) != NULL) {
		if (strlen(buf) < 1)
			continue;

		if (buf[0] == '#') {
			if (strlen(buf) < 3)
				continue;

			/*
			 * Note the '@' flag was used only in the 2006
			 * table; previious to that the flag was '$'.
			 */
			if (buf[1] == '@' || buf[1] == '$') {
				if (sscanf(&buf[2], "%lu", &expire) !=
				    1)
					return (-1);

				continue;
			}
		}
		if (sscanf(buf, "%lu %d", &leap, &offset) == 2) {

			/*
			 * Valid offsets must increase by one for each
			 * leap.
			 */
			if (i++ != offset)
				return (-1);
		}
	}

	/*
	 * There must be at least one leap.
	 */
	if (i == 10)
		return (-1);

	leap_tai = offset;
	leap_sec = leap;
	leap_expire = expire;
	return (0);
}


/*
 * leap_month - returns seconds until the end of the month.
 */
u_long
leap_month(
	u_long	sec		/* current NTP second */
	)
{
	int	     leap;
	int32	     year, month;
	u_int32	     ndays;
	ntpcal_split tmp;
	vint64	     tvl;

	/* --*-- expand time and split to days */
	tvl   = ntpcal_ntp_to_ntp(sec, NULL);
	tmp   = ntpcal_daysplit(&tvl);
	/* --*-- split to years and days in year */
	tmp   = ntpcal_split_eradays(tmp.hi + DAY_NTP_STARTS - 1, &leap);
	year  = tmp.hi;
	/* --*-- split days of year to month */
	tmp   = ntpcal_split_yeardays(tmp.lo, leap);
	month = tmp.hi;
	/* --*-- get nominal start of next month */
	ndays = ntpcal_edate_to_eradays(year, month+1, 0) + 1 - DAY_NTP_STARTS;
	
	return (u_int32)(ndays*SECSPERDAY - sec);
}


/*
 * getauthkeys - read the authentication keys from the specified file
 */
void
getauthkeys(
	const char *keyfile
	)
{
	int len;

	len = strlen(keyfile);
	if (!len)
		return;
	
#ifndef SYS_WINNT
	key_file_name = erealloc(key_file_name, len + 1);
	memcpy(key_file_name, keyfile, len + 1);
#else
	key_file_name = erealloc(key_file_name, _MAX_PATH);
	if (len + 1 > _MAX_PATH)
		return;
	if (!ExpandEnvironmentStrings(keyfile, key_file_name,
				      _MAX_PATH)) {
		msyslog(LOG_ERR,
			"ExpandEnvironmentStrings(KEY_FILE) failed: %m");
		strncpy(key_file_name, keyfile, _MAX_PATH);
	}
#endif /* SYS_WINNT */

	authreadkeys(key_file_name);
}


/*
 * rereadkeys - read the authentication key file over again.
 */
void
rereadkeys(void)
{
	if (NULL != key_file_name)
		authreadkeys(key_file_name);
}


#if notyet
/*
 * ntp_exit - document explicitly that ntpd has exited
 */
void
ntp_exit(int retval)
{
	msyslog(LOG_ERR, "EXITING with return code %d", retval);
	exit(retval);
}
#endif

/*
 * fstostr - prettyprint NTP seconds
 */
char * fstostr(
	time_t	ntp_stamp
	)
{
	char	*	buf;
	struct tm *	tm;
	time_t		unix_stamp;

	LIB_GETBUF(buf);
	unix_stamp = ntp_stamp - JAN_1970;
	tm = gmtime(&unix_stamp);
	if (NULL == tm)
#ifdef WAIT_FOR_NTP_CRYPTO_C_CALLERS_ABLE_TO_HANDLE_MORE_THAN_20_CHARS
		msnprintf(buf, LIB_BUFLENGTH, "gmtime: %m");
#else
		strncpy(buf, "gmtime() error", LIB_BUFLENGTH);
#endif
	else
		snprintf(buf, LIB_BUFLENGTH, "%04d%02d%02d%02d%02d",
			 tm->tm_year + 1900, tm->tm_mon + 1,
			 tm->tm_mday, tm->tm_hour, tm->tm_min);

	return buf;
}


/*
 * ntpd_time_stepped is called back by step_systime(), allowing ntpd
 * to do any one-time processing necessitated by the step.
 */
void
ntpd_time_stepped(void)
{
	u_int saved_mon_enabled;

	/*
	 * flush the monitor MRU list which contains l_fp timestamps
	 * which should not be compared across the step.
	 */
	if (MON_OFF != mon_enabled) {
		saved_mon_enabled = mon_enabled;
		mon_stop(MON_OFF);
		mon_start(saved_mon_enabled);
	}

	/* inform interpolating Windows code to allow time to go back */
#ifdef SYS_WINNT
	win_time_stepped();
#endif
}
