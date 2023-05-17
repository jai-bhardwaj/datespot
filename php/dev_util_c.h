#ifndef DUC_ERROR_H
#define DUC_ERROR_H

/**
 * Retrieves the error message associated with the given error code.
 *
 * @param code The error code for which the error message is requested.
 * @return A pointer to the error message string.
 *
 * @note The caller is responsible for freeing the memory allocated for the error message.
 */
char *duc_getErrorMsg(int code);

#endif /* DUC_ERROR_H */