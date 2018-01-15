import sys, os


def send_to_slack(title, message):
    import requests
    user_name = title

    token = 'xoxp-96852521365-96857869169-96865466370-fefad75e3992481bc67a3d043efb5b22'
    # token = "xoxp-35520343894-36865404167-96910752627-34c2ffe5daa7f71bfc900a25cf9578e2"
    channel = "@fukuta6140"
    bot_icon = "http://i.imgur.com/DP2oyoM.jpg"

    post_url = 'https://slack.com/api/chat.postMessage'

    payload = {
        'token': token,
        'channel': channel,
        'username': user_name,
        'icon_url': bot_icon,
        'text': message,
    }

    requests.post(post_url, data=payload)


def set_debugger(send_email=True, error_func=None):
    if hasattr(sys.excepthook, '__name__') and sys.excepthook.__name__ in ['excepthook', 'apport_excepthook']:
        from IPython.core import ultratb

        class MyTB(ultratb.FormattedTB):
            def __init__(self, mode='Plain', color_scheme='Linux', call_pdb=False,
                         ostream=None,
                         tb_offset=0, long_header=False, include_vars=False,
                         check_cache=None, send_email=False, error_func=None):
                self.send_email = send_email
                self.color_scheme = color_scheme
                self.error_func = error_func
                ultratb.FormattedTB.__init__(self, mode=mode,
                                             color_scheme=color_scheme,
                                             call_pdb=call_pdb,
                                             ostream=ostream,
                                             tb_offset=tb_offset,
                                             long_header=long_header,
                                             include_vars=include_vars,
                                             check_cache=check_cache)

            def __call__(self, etype=None, evalue=None, etb=None):
                if self.send_email and not etype.__name__ == 'KeyboardInterrupt':
                    try:
                        title = os.uname()[1]
                        self.set_colors('NoColor')
                        body = ' '.join(sys.argv) + '\n'
                        body += self.text(etype, evalue, etb)
                        self.set_colors(self.color_scheme)
                        send_to_slack(title, body)
                    except Exception as e:
                        print(e)
                        pass

                if etype.__name__ == 'KeyboardInterrupt':
                    return
                else:
                    ultratb.FormattedTB.__call__(self, etype=etype, evalue=evalue, etb=etb)

                if self.error_func is not None:
                    try:
                        self.error_func()
                    except Exception as e:
                        print(e)
                        pass

        sys.excepthook = MyTB(call_pdb=True, send_email=send_email, error_func=error_func)
        print('debugger has been set')


set_debugger()
