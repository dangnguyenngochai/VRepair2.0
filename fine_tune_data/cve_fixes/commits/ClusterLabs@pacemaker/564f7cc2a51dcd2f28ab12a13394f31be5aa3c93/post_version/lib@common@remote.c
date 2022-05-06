/*
 * Copyright (c) 2008 Andrew Beekhof
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#include <crm_internal.h>
#include <crm/crm.h>

#include <sys/param.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netdb.h>


#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <glib.h>

#include <crm/common/ipc.h>
#include <crm/common/xml.h>

#ifdef HAVE_GNUTLS_GNUTLS_H
#  undef KEYFILE
#  include <gnutls/gnutls.h>
#endif

#ifdef HAVE_GNUTLS_GNUTLS_H
const int anon_tls_kx_order[] = {
    GNUTLS_KX_ANON_DH,
    GNUTLS_KX_DHE_RSA,
    GNUTLS_KX_DHE_DSS,
    GNUTLS_KX_RSA,
    0
};

int
crm_initiate_client_tls_handshake(void *session_data, int timeout_ms)
{
    int rc = 0;
    int pollrc = 0;
    time_t start = time(NULL);
    gnutls_session *session = session_data;

    do {
        rc = gnutls_handshake(*session);
        if (rc == GNUTLS_E_INTERRUPTED || rc == GNUTLS_E_AGAIN) {
            pollrc = crm_recv_remote_ready(session, TRUE, 1000);
            if (pollrc < 0) {
                /* poll returned error, there is no hope */
                rc = -1;
            }
        }
    } while (((time(NULL) - start) < (timeout_ms/1000)) &&
            (rc == GNUTLS_E_INTERRUPTED || rc == GNUTLS_E_AGAIN));

    return rc;
}

void *
crm_create_anon_tls_session(int csock, int type /* GNUTLS_SERVER, GNUTLS_CLIENT */, void *credentials)
{
    gnutls_session *session = gnutls_malloc(sizeof(gnutls_session));

    gnutls_init(session, type);
#  ifdef HAVE_GNUTLS_PRIORITY_SET_DIRECT
/*      http://www.manpagez.com/info/gnutls/gnutls-2.10.4/gnutls_81.php#Echo-Server-with-anonymous-authentication */
    gnutls_priority_set_direct(*session, "NORMAL:+ANON-DH", NULL);
/*	gnutls_priority_set_direct (*session, "NONE:+VERS-TLS-ALL:+CIPHER-ALL:+MAC-ALL:+SIGN-ALL:+COMP-ALL:+ANON-DH", NULL); */
#  else
    gnutls_set_default_priority(*session);
    gnutls_kx_set_priority(*session, anon_tls_kx_order);
#  endif
    gnutls_transport_set_ptr(*session, (gnutls_transport_ptr) GINT_TO_POINTER(csock));
    switch (type) {
    case GNUTLS_SERVER:
        gnutls_credentials_set(*session, GNUTLS_CRD_ANON, (gnutls_anon_server_credentials_t) credentials);
        break;
    case GNUTLS_CLIENT:
        gnutls_credentials_set(*session, GNUTLS_CRD_ANON, (gnutls_anon_client_credentials_t) credentials);
        break;
    }

    return session;
}

static int
crm_send_tls(gnutls_session * session, const char *buf, size_t len)
{
    const char *unsent = buf;
    int rc = 0;
    int total_send;

    if (buf == NULL) {
        return -1;
    }

    total_send = len;
    crm_trace("Message size: %d", len);

    while (TRUE) {
        rc = gnutls_record_send(*session, unsent, len);

        if (rc == GNUTLS_E_INTERRUPTED || rc == GNUTLS_E_AGAIN) {
            crm_debug("Retry");

        } else if (rc < 0) {
            crm_err("Connection terminated rc = %d", rc);
            break;

        } else if (rc < len) {
            crm_debug("Only sent %d of %d bytes", rc, len);
            len -= rc;
            unsent += rc;
        } else {
            crm_debug("Sent %d bytes", rc);
            break;
        }
    }

    return rc < 0 ? rc : total_send;
}


/*!
 * \internal
 * \brief Read bytes off non blocking tls session.
 *
 * \param session - tls session to read
 * \param max_size - max bytes allowed to read for buffer. 0 assumes no limit
 *
 * \note only use with NON-Blocking sockets. Should only be used after polling socket.
 *       This function will return once max_size is met, the socket read buffer
 *       is empty, or an error is encountered.
 *
 * \retval '\0' terminated buffer on success
 */
static char *
crm_recv_tls(gnutls_session * session, size_t max_size, size_t *recv_len, int *disconnected)
{
    char *buf = NULL;
    int rc = 0;
    size_t len = 0;
    size_t chunk_size = max_size ? max_size : 1024;
    size_t buf_size = 0;
    size_t read_size = 0;

    if (session == NULL) {
        if (disconnected) {
            *disconnected = 1;
        }
        goto done;
    }

    buf = calloc(1, chunk_size + 1);
    buf_size = chunk_size;

    while (TRUE) {
        read_size = buf_size - len;

        /* automatically grow the buffer when needed if max_size is not set.*/
        if (!max_size && (read_size < (chunk_size / 2))) {
            buf_size += chunk_size;
            crm_trace("Grow buffer by %d more bytes. buf is now %d bytes", (int)chunk_size, buf_size);
            buf = realloc(buf, buf_size + 1);
            CRM_ASSERT(buf != NULL);

            read_size = buf_size - len;
        }

        rc = gnutls_record_recv(*session, buf + len, read_size);

        if (rc > 0) {
            crm_trace("Got %d more bytes.", rc);
            len += rc;
            /* always null terminate buffer, the +1 to alloc always allows for this.*/
            buf[len] = '\0';
        }
        if (max_size && (max_size == read_size)) {
            crm_trace("Buffer max read size %d met" , max_size);
            goto done;
        }

        /* process any errors. */
        if (rc == GNUTLS_E_INTERRUPTED) {
            crm_trace("EINTR encoutered, retry tls read");
        } else if (rc == GNUTLS_E_AGAIN) {
            crm_trace("non-blocking, exiting read on rc = %d", rc);
            goto done;
        } else if (rc <= 0) {
            if (rc == 0) {
                crm_debug("EOF encoutered during TLS read");
            } else {
                crm_debug("Error receiving message: %s (%d)", gnutls_strerror(rc), rc);
            }
            if (disconnected) {
                *disconnected = 1;
            }
            goto done;
        }
    }

done:
    if (recv_len) {
        *recv_len = len;
    }
    if (!len) {
        free(buf);
        buf = NULL;
    }
    return buf;

}
#endif

static int
crm_send_plaintext(int sock, const char *buf, size_t len)
{

    int rc = 0;
    const char *unsent = buf;
    int total_send;

    if (buf == NULL) {
        return -1;
    }
    total_send = len;

    crm_trace("Message on socket %d: size=%d", sock, len);
  retry:
    rc = write(sock, unsent, len);
    if (rc < 0) {
        switch (errno) {
        case EINTR:
        case EAGAIN:
            crm_trace("Retry");
            goto retry;
        default:
            crm_perror(LOG_ERR, "Could only write %d of the remaining %d bytes", rc, (int) len);
            break;
        }

    } else if (rc < len) {
        crm_trace("Only sent %d of %d remaining bytes", rc, len);
        len -= rc;
        unsent += rc;
        goto retry;

     } else {
        crm_trace("Sent %d bytes: %.100s", rc, buf);
    }

    return rc < 0 ? rc : total_send;

}

/*!
 * \internal
 * \brief Read bytes off non blocking socket.
 *
 * \param session - tls session to read
 * \param max_size - max bytes allowed to read for buffer. 0 assumes no limit
 *
 * \note only use with NON-Blocking sockets. Should only be used after polling socket.
 *       This function will return once max_size is met, the socket read buffer
 *       is empty, or an error is encountered.
 *
 * \retval '\0' terminated buffer on success
 */
static char *
crm_recv_plaintext(int sock, size_t max_size, size_t *recv_len, int *disconnected)
{
    char *buf = NULL;
    ssize_t rc = 0;
    ssize_t len = 0;
    ssize_t chunk_size = max_size ? max_size : 1024;
    size_t buf_size = 0;
    size_t read_size = 0;

    if (sock <= 0) {
        if (disconnected) {
            *disconnected = 1;
        }
        goto done;
    }

    buf = calloc(1, chunk_size + 1);
    buf_size = chunk_size;

    while (TRUE) {
        errno = 0;
        read_size = buf_size - len;

        /* automatically grow the buffer when needed if max_size is not set.*/
        if (!max_size && (read_size < (chunk_size / 2))) {
            buf_size += chunk_size;
            crm_trace("Grow buffer by %d more bytes. buf is now %d bytes", (int)chunk_size, buf_size);
            buf = realloc(buf, buf_size + 1);
            CRM_ASSERT(buf != NULL);

            read_size = buf_size - len;
        }

        rc = read(sock, buf + len, chunk_size);

        if (rc > 0) {
            crm_trace("Got %d more bytes. errno=%d", (int)rc, errno);
            len += rc;
            /* always null terminate buffer, the +1 to alloc always allows for this.*/
            buf[len] = '\0';
        }
        if (max_size && (max_size == read_size)) {
            crm_trace("Buffer max read size %d met" , max_size);
            goto done;
        }

        if (rc > 0) {
            continue;
        } else if (rc == 0) {
            if (disconnected) {
                *disconnected = 1;
            }
            crm_trace("EOF encoutered during read");
            goto done;
        }

        /* process errors */
        if (errno == EINTR) {
            crm_trace("EINTER encoutered, retry socket read.");
        } else if (errno == EAGAIN) {
            crm_trace("non-blocking, exiting read on rc = %d", rc);
            goto done;
        } else if (errno <= 0) {
            if (disconnected) {
                *disconnected = 1;
            }
            crm_debug("Error receiving message: %d", (int)rc);
            goto done;
        }
    }

done:
    if (recv_len) {
        *recv_len = len;
    }
    if (!len) {
        free(buf);
        buf = NULL;
    }
    return buf;
}

static int
crm_send_remote_msg_raw(void *session, const char *buf, size_t len, gboolean encrypted)
{
    int rc = -1;
    if (encrypted) {
#ifdef HAVE_GNUTLS_GNUTLS_H
        rc = crm_send_tls(session, buf, len);
#else
        CRM_ASSERT(encrypted == FALSE);
#endif
    } else {
        rc = crm_send_plaintext(GPOINTER_TO_INT(session), buf, len);
    }
    return rc;
}

int
crm_send_remote_msg(void *session, xmlNode * msg, gboolean encrypted)
{
    int rc = -1;
    char *xml_text = NULL;
    int len = 0;

    xml_text = dump_xml_unformatted(msg);
    if (xml_text) {
        len = strlen(xml_text);
    } else {
        crm_err("Invalid XML, can not send msg");
        return -1;
    }

    rc = crm_send_remote_msg_raw(session, xml_text, len, encrypted);
    if (rc < 0) {
        goto done;
    }
    rc = crm_send_remote_msg_raw(session, REMOTE_MSG_TERMINATOR, strlen(REMOTE_MSG_TERMINATOR), encrypted);

done:
    if (rc < 0) {
        crm_err("Failed to send remote msg, rc = %d", rc);
    }

    free(xml_text);
    return rc;
}

/*!
 * \internal
 * \brief handles the recv buffer and parsing out msgs.
 * \note new_data is owned by this function once it is passed in.
 */
xmlNode *
crm_parse_remote_buffer(char **msg_buf)
{
    char *buf = NULL;
    char *start = NULL;
    char *end = NULL;
    xmlNode *xml = NULL;

    if (*msg_buf == NULL) {
        return NULL;
    }

    /* take ownership of the buffer */
    buf = *msg_buf;
    *msg_buf = NULL;

    /* MSGS are separated by a '\r\n\r\n'. Split a message off the buffer and return it. */
    start = buf;
    end = strstr(start, REMOTE_MSG_TERMINATOR);

    while (!xml && end) {

        /* grab the message */
        end[0] = '\0';
        end += strlen(REMOTE_MSG_TERMINATOR);

        xml = string2xml(start);
        if (xml == NULL) {
            crm_err("Couldn't parse: '%.120s'", start);
        }
        start = end;
        end = strstr(start, REMOTE_MSG_TERMINATOR);
    }

    if (xml && start) {
        /* we have msgs left over, save it until next time */
        *msg_buf = strdup(start);
        free(buf);
    } else if (!xml) {
        /* no msg present */
        *msg_buf = buf;
    }

    return xml;
}

/*!
 * \internal
 * \brief Determine if a remote session has data to read
 *
 * \retval 0, timeout occured.
 * \retval positive, data is ready to be read
 * \retval negative, session has ended
 */
int
crm_recv_remote_ready(void *session, gboolean encrypted, int timeout /* ms */)
{
    struct pollfd fds = { 0, };
    int sock = 0;
    void *sock_ptr = NULL;
    int rc = 0;
    time_t start;

    if (encrypted) {
#ifdef HAVE_GNUTLS_GNUTLS_H
        gnutls_session *tls_session = session;
        sock_ptr = gnutls_transport_get_ptr(*tls_session);
#else
        CRM_ASSERT(encrypted == FALSE);
#endif
    } else {
        sock_ptr = session;
    }

    sock = GPOINTER_TO_INT(sock_ptr);
    if (sock <= 0) {
        return -ENOTCONN;
    }

    start = time(NULL);
    errno = 0;
    do {
        fds.fd = sock;
        fds.events = POLLIN;

        /* If we got an EINTR while polling, and we have a
         * specific timeout we are trying to honor, attempt
         * to adjust the timeout to the closest second. */
        if (errno == EINTR && (timeout > 0)) {
            timeout = timeout - ((time(NULL) - start) * 1000);
            if (timeout < 1000) {
                timeout = 1000;
            }
        }

        rc = poll(&fds, 1, timeout);
    } while (rc < 0 && errno == EINTR);

    return rc;
}

char *
crm_recv_remote_raw(void *session, gboolean encrypted, size_t max_recv, size_t *recv_len, int *disconnected)
{
    char *reply = NULL;
    if (recv_len) {
        *recv_len = 0;
    }

    if (disconnected) {
        *disconnected = 0;
    }

    if (encrypted) {
#ifdef HAVE_GNUTLS_GNUTLS_H
        reply = crm_recv_tls(session, max_recv, recv_len, disconnected);
#else
        CRM_ASSERT(encrypted == FALSE);
#endif
    } else {
        reply = crm_recv_plaintext(GPOINTER_TO_INT(session), max_recv, recv_len, disconnected);
    }
    if (reply == NULL || strlen(reply) == 0) {
        crm_trace("Empty reply");
    }

    return reply;
}

/*!
 * \internal
 * \brief Read data off the socket until at least one full message is present or timeout occures.
 * \retval TRUE message read
 * \retval FALSE full message not read
 */

gboolean
crm_recv_remote_msg(void *session, char **recv_buf, gboolean encrypted, int total_timeout /*ms */, int *disconnected)
{
    int ret;
    size_t request_len = 0;
    time_t start = time(NULL);
    char *raw_request = NULL;
    int remaining_timeout = 0;

    if (total_timeout == 0) {
        total_timeout = 10000;
    } else if (total_timeout < 0) {
        total_timeout = 60000;
    }
    *disconnected = 0;

    remaining_timeout = total_timeout;
    while ((remaining_timeout > 0) && !(*disconnected)) {

        /* read some more off the tls buffer if we still have time left. */
        crm_trace("waiting to receive remote msg, starting timeout %d, remaining_timeout %d", total_timeout, remaining_timeout);
        ret = crm_recv_remote_ready(session, encrypted, remaining_timeout);
        raw_request = NULL;

        if (ret == 0) {
            crm_err("poll timed out (%d ms) while waiting to receive msg", remaining_timeout);
            return FALSE;

        } else if (ret < 0) {
            if (errno != EINTR) {
                crm_debug("poll returned error while waiting for msg, rc: %d, errno: %d", ret, errno);
                *disconnected = 1;
                return FALSE;
            }
            crm_debug("poll EINTR encountered during poll, retrying");
        } else {
            raw_request = crm_recv_remote_raw(session, encrypted, 0, &request_len, disconnected);
        }

        remaining_timeout = remaining_timeout - ((time(NULL) - start) * 1000);

        if (!raw_request) {
            crm_debug("Empty msg received after poll");
            continue;
        }

        if (*recv_buf) {
            int old_len = strlen(*recv_buf);

            crm_trace("Expanding recv buffer from %d to %d", old_len, old_len+request_len);

            *recv_buf = realloc(*recv_buf, old_len + request_len + 1);
            memcpy(*recv_buf + old_len, raw_request, request_len);
            *(*recv_buf+old_len+request_len) = '\0';
            free(raw_request);
        } else {
            *recv_buf = raw_request;
        }

        if (strstr(*recv_buf, REMOTE_MSG_TERMINATOR)) {
            return TRUE;
        }
    }

    return FALSE;
}

/*!
 * \internal
 * \brief tcp connection to server at specified port
 * \retval positive, socket fd.
 * \retval negative, failed to connect.
 */
int
crm_remote_tcp_connect(const char *host, int port)
{
    struct addrinfo *res;
    struct addrinfo *rp;
    struct addrinfo hints;
    const char *server = host;
    int ret_ga;
    int sock;

    /* getaddrinfo */
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_CANONNAME;

    crm_debug("Looking up %s", server);
    ret_ga = getaddrinfo(server, NULL, &hints, &res);
    if (ret_ga) {
        crm_err("getaddrinfo: %s", gai_strerror(ret_ga));
        return -1;
    }

    if (!res || !res->ai_addr) {
        crm_err("getaddrinfo failed");
        return -1;
    }

    for (rp = res; rp != NULL; rp = rp->ai_next) {
        struct sockaddr *addr = rp->ai_addr;
        int flag = 0;
        if (!addr) {
            continue;
        }

        if (rp->ai_canonname) {
            server = res->ai_canonname;
        }
        crm_debug("Got address %s for %s", server, host);

        /* create socket */
        sock = socket(rp->ai_family, SOCK_STREAM, IPPROTO_TCP);
        if (sock == -1) {
            crm_err("Socket creation failed for remote client connection.");
            continue;
        }
        if (addr->sa_family == AF_INET6) {
            struct sockaddr_in6 *addr_in = (struct sockaddr_in6 *) addr;
            addr_in->sin6_port = htons(port);
        } else {
            struct sockaddr_in *addr_in = (struct sockaddr_in *) addr;
            addr_in->sin_port = htons(port);
            crm_info("Attempting to connect to remote server at %s:%d", inet_ntoa(addr_in->sin_addr), port);
        }

        if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
            if ((flag = fcntl(sock, F_GETFL)) >= 0) {
                if (fcntl(sock, F_SETFL, flag | O_NONBLOCK) < 0) {
                    crm_err( "fcntl() write failed");
                    close(sock);
                    sock = -1;
                    continue;
                }
            }
            break;                  /* Success */
        }

        close(sock);
        sock = -1;
    }
    freeaddrinfo(res);

    return sock;
}

