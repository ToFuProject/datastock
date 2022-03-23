

import warnings


import numpy as np
import matplotlib.pyplot as plt


from . import _generic_check
from ._class1 import *
from . import _class2_interactivity
from . import _class1_compute


# #################################################################
# #################################################################
#               Main class
# #################################################################


class DataStock2(DataStock1):
    """ Handles matplotlib interactivity """

    _LPAXES = ['axes', 'type']
    __store_rcParams = None

    # ----------------------
    #   Add objects
    # ----------------------

    def add_mobile(
        self,
        handle=None,
        key=None,
        ref=None,
        data=None,
        dtype=None,
        bstr=None,
        visible=None,
        group_vis=None,
        axes=None,
        **kwdargs,
    ):

        # ----------
        # check ref

        if isinstance(ref, str):
            ref = (ref,)
        if isinstance(ref, list):
            ref = tuple(ref)

        if ref is None or not all([rr in self._dref.keys() for rr in ref]):
            msg = (
                "Arg ref must be a tuple of existing ref keys!\n"
                f"\t- Provided: {ref}"
            )
            raise Exception(msg)
        nref = len(ref)

        # ----------
        # check dtype

        if isinstance(dtype, str):
            dtype = [dtype]
        dtype = _generic_check._check_var_iter(
            dtype,
            'dtype',
            types=list,
            types_iter=str,
            allowed=['xdata', 'ydata', 'data', 'data.T', 'alpha', 'txt']
        )
        if len(dtype) != nref:
            msg = (
                f"For mobile {key}:\n"
                "Arg dtype must be a list, the same length as ref!\n"
                f"\t- dtype: {dtype}\n"
                f"\t- ref: {ref}\n"
            )
            raise Exception(msg)

        # ----------
        # check data

        if isinstance(data, str):
            data = (data,)
        if isinstance(data, list):
            data = tuple(data)
        if data is None:
            data = ['index' for rr in ref]

        c0 = (
            nref == len(data)
            and all([rr == 'index' or rr in self._ddata.keys() for rr in data])
        )
        if not c0:
            msg = (
                "Arg data must be a tuple of existing data keys!\n"
                "It should have the same length as ref!\n"
                f"\t- Provided ref: {ref}\n"
                f"\t- Provided data: {data}"
            )
            raise Exception(msg)

        # ------------------
        # check axis vs data

        axis = [
            0 if dd == 'index'
            else self._ddata[dd]['ref'].index(ref[ii])
            for ii, dd in enumerate(data)
        ]

        # ---------------------
        # check ndata vs ndtype

        if len(set(dtype)) != len(set(data)):
            if all([dd == 'index' for dd in data]):
                pass
            else:
                msg = (
                    f"In dmobile['{key}']:\n"
                    "Nb. of different dtypes must match nb of different data!\n"
                    f"\t- dtype: {dtype}\n"
                    f"\t- data: {data}\n"
                )
                raise Exception(msg)

        super().add_obj(
            which='mobile',
            key=key,
            handle=handle,
            group=None,
            group_vis=group_vis,
            ref=ref,
            data=data,
            axis=axis,
            dtype=dtype,
            visible=visible,
            bstr=bstr,
            axes=axes,
            func=None,
            **kwdargs,
        )

    def add_axes(
        self,
        handle=None,
        key=None,
        type=None,
        refx=None,
        refy=None,
        datax=None,
        datay=None,
        invertx=None,
        inverty=None,
        harmonize=None,
        **kwdargs,
    ):

        # ----------------
        # check refx, refy

        # if refx is None and refy is None:
            # msg = f"Please provide at least refx or refy for axes {key}!"
            # raise Exception(msg)

        if isinstance(refx, str):
            refx = [refx]
        if isinstance(refy, str):
            refy = [refy]

        c0 =(
            isinstance(refx, list)
            and all([rr in self._dref.keys() for rr in refx])
        )
        if refx is not None and not c0:
            msg = "Arg refx must be a list of valid ref keys!"
            raise Exception(msg)

        c0 =(
            isinstance(refy, list)
            and all([rr in self._dref.keys() for rr in refy])
        )
        if refy is not None and not c0:
            msg = "Arg refy must be a list of valid ref keys!"
            raise Exception(msg)

        # data
        if isinstance(datax, str):
            datax = [datax]
        if isinstance(datay, str):
            datay = [datay]
        if datax is None and refx is not None:
            datax = ['index' for rr in refx]
        if datay is None and refy is not None:
            datay = ['index' for rr in refy]

        # ref vs data
        if refx is not None and len(refx) != len(datax):
            msg = (
                "refx and datax must have the same length!\n"
                f"\t- refx: {refx}"
                f"\t- datax: {datax}"
            )
            raise Exception(msg)
        if refy is not None and len(refy) != len(datay):
            msg = (
                "refy and datay must have the same length!\n"
                f"\t- refx: {refy}"
                f"\t- datax: {datay}"
            )
            raise Exception(msg)

        super().add_obj(
            which='axes',
            key=key,
            handle=handle,
            type=type,
            groupx=None,
            groupy=None,
            refx=refx,
            refy=refy,
            datax=datax,
            datay=datay,
            invertx=invertx,
            inverty=inverty,
            bck=None,
            mobile=None,
            canvas=None,
            harmonize=harmonize,
            **kwdargs,
        )

        # add canvas if not already stored
        if 'canvas' not in self._dobj.keys():
            self.add_canvas(
                handle=handle.figure.canvas,
                harmonize=harmonize,
            )
        else:
            lisin = [
                k0 for k0, v0 in self._dobj['canvas'].items()
                if v0['handle'] == handle.figure.canvas
            ]
            if len(lisin) == 0:
                self.add_canvas(
                    handle=handle.figure.canvas,
                    harmonize=harmonize,
                )

    def add_canvas(self, key=None, handle=None, harmonize=None):
        """ Add canvas and interactivity obj """
        interactive = (
            hasattr(handle, 'toolbar')
            and handle.toolbar is not None
        )
        self.add_obj(
            which='canvas',
            key=key,
            handle=handle,
            interactive=interactive,
            harmonize=harmonize,
        )

    # ------------------
    # Properties
    # ------------------

    @property
    def dax(self):
        return self.dobj.get('axes', {})

    @property
    def dinteractivity(self):
        return self._dinteractivity

    # ------------------
    # Debug mode
    # ------------------

    def set_debug(self, debug=None):
        """ Set debug mode to True / False """
        debug = _generic_check._check_var(
            debug,
            'debug',
            default=False,
            types=bool,
        )
        self.debug = debug

    def show_debug(self):
        """ Display information relevant for live debugging """
        print('\n\n')
        return self.show(show_which=['ref', 'group', 'interactivity'])

    # ------------------
    # Setup interactivity
    # ------------------

    def setup_interactivity(
        self,
        kinter=None,
        dgroup=None,
        dkeys=None,
        dinc=None,
        cur_ax=None,
        debug=None,
    ):
        """

        dgroup = {
            'group0': {
                'ref': ['ref0', 'ref1', ...],
                'nmax': 3,
                'colors': ['r', 'g', 'b'],
            }
        }

        """

        # ----------
        # Check dgroup
        dgroup, newgroup = _class2_interactivity._setup_dgroup(
            dgroup=dgroup,
            dobj0=self._dobj,
            dref0=self._dref,
        )

        # ----------
        # Check increment dict

        dinc, newinc = _class2_interactivity._setup_dinc(
            dinc=dinc,
            lparam_ref=self.get_lparam(which='ref'),
            dref0=self._dref,
        )

        # ----------------------------------------------------------
        # make sure all refs are known and are associated to a group

        drefgroup = _class2_interactivity._setup_drefgroup(
            dref0=self._dref,
            dgroup=dgroup,
        )

        #  add indices to ref
        for k0, v0 in self._dref.items():
            if drefgroup[k0] is not None:
                self.add_indices_per_ref(
                    indices=np.zeros((dgroup[drefgroup[k0]]['nmax'],), dtype=int),
                    ref=k0,
                    distribute=False,
                )

        # --------------------------------------
        # update dax with groupx, groupy and inc

        daxgroupx = dict.fromkeys(self._dobj['axes'].keys())
        daxgroupy = dict.fromkeys(self._dobj['axes'].keys())
        dinc_axes = dict.fromkeys(self._dobj['axes'].keys())
        for k0, v0 in self._dobj['axes'].items():
            if v0['refx'] is None:
                daxgroupx[k0] = None
            else:
                daxgroupx[k0] = [drefgroup[k1] for k1 in v0['refx']]
            if v0['refy'] is None:
                daxgroupy[k0] = None
            else:
                daxgroupy[k0] = [drefgroup[k1] for k1 in v0['refy']]

            # increment
            dinc_axes[k0] = {
                'left': -1, 'right': 1,
                'down': -1, 'up': 1,
            }

        # -------
        # dgroup

        # update with axes
        for k0, v0 in dgroup.items():
            lkax = [
                k1 for k1, v1 in self._dobj['axes'].items()
                if (daxgroupx[k1] is not None and k0 in daxgroupx[k1])
                or (daxgroupy[k1] is not None and k0 in daxgroupy[k1])
            ]
            dgroup[k0]['axes'] = lkax

        for ii, (k0, v0) in enumerate(dgroup.items()):
            harmonize = ii == len(dgroup) - 1
            self.add_obj(
                which='group',
                key=k0,
                harmonize=harmonize,
                **v0,
            )

        # -----------------------
        # Populate new parameters

        self.set_param(which='axes', param='groupx', value=daxgroupx)
        self.set_param(which='axes', param='groupy', value=daxgroupy)
        self.add_param(which='axes', param='inc', value=dinc_axes)

        self.add_param(which='ref', param='group', value=drefgroup)
        self.add_param(which='ref', param='inc', value=dinc)

        # --------------------------
        # update mobile with group, group_vis and func

        _class2_interactivity._setup_mobile(
            dmobile=self._dobj['mobile'],
            dref=self._dref,
            ddata=self._ddata,
        )

        # --------------------
        # axes mobile, refs and canvas

        daxcan = dict.fromkeys(self._dobj['axes'].keys())
        for k0, v0 in self._dobj['axes'].items():

            # Update mobile
            self._dobj['axes'][k0]['mobile'] = [
                k1 for k1, v1 in self._dobj['mobile'].items()
                if v1['axes'] == k0
            ]

            # ref
            if v0['refx'] is not None:
                for ii, rr in enumerate(v0['refx']):
                    if v0['datax'][ii] is None:
                        self._dobj['axes'][k0]['datax'][ii] = 'index'

            # canvas
            lcan = [
                k1 for k1, v1 in self._dobj['canvas'].items()
                if v1['handle'] == v0['handle'].figure.canvas
            ]
            assert len(lcan) == 1
            self._dobj['axes'][k0]['canvas'] = lcan[0]

        # ---------
        # dkeys

        dkeys = _class2_interactivity._setup_keys(dkeys=dkeys, dgroup=dgroup)

        # implement dict
        for ii, (k0, v0) in enumerate(dkeys.items()):
            harmonize = ii == len(dgroup) - 1
            self.add_obj(
                which='key',
                key=k0,
                harmonize=harmonize,
                **v0,
            )

        lact = set([v0['action'] for v0 in dkeys.values()])
        self.__dkeys_r = {
            k0: [k1 for k1 in dkeys.keys() if dkeys[k1]['action'] == k0]
            for k0 in lact
        }

        # ---------
        # dinter

        dinter = {
            'cur_ax': cur_ax,
            'cur_ax_panzoom': cur_ax,
            'cur_groupx': None,
            'cur_groupy': None,
            'cur_refx': None,
            'cur_refy': None,
            'cur_datax': None,
            'cur_datay': None,
            'follow': True,
        }

        if kinter is None:
            if hasattr(self, 'kinter'):
                kinter = self.kinter
            else:
                kinter = 'inter0'
        self.kinter = kinter
        self.add_obj(
            which='interactivity',
            key=kinter,
            **dinter,
        )

        _class2_interactivity._set_dbck(
            lax=self._dobj['axes'].keys(),
            daxes=self._dobj['axes'],
            dcanvas=self._dobj['canvas'],
            dmobile=self._dobj['mobile'],
        )

        # -----------------------------------
        # set current axe / group / ref / data...

        if cur_ax is None:
            cur_ax = [
                k0 for k0, v0 in self._dobj['axes'].items()
                if v0['groupx'] is not None and v0['groupy'] is not None
            ]
            if len(cur_ax) == 0:
                cur_ax = list(self._dobj['axes'].keys())[0]
            else:
                cur_ax = cur_ax[0]

        self._get_current_grouprefdata_from_kax(kax=cur_ax)
        self.set_debug(debug)

    # ----------------------------
    # Ensure connectivity possible
    # ----------------------------

    def _warn_ifnotInteractive(self):
        warn = False

        c0 = (
            len(self._dobj.get('axes', {})) > 0
            and len(self._dobj.get('canvas', {})) > 0
        )
        if c0:
            dcout = {
                k0: v0['handle'].__class__.__name__
                for k0, v0 in self._dobj['canvas'].items()
                if not v0['interactive']
            }
            if len(dcout) > 0:
                lstr = '\n'.join(
                    [f'\t- {k0}: {v0}' for k0, v0 in dcout.items()]
                )
                msg = (
                    "Non-interactive backends identified (prefer Qt5Agg):\n"
                    f"\t- backend : {plt.get_backend()}\n"
                    f"\t- canvas  :\n{lstr}"
                )
                warn = True
        else:
            msg = ("No available axes / canvas for interactivity")
            warn = True

        # raise warning
        if warn:
            warnings.warn(msg)

        return warn

    # ----------------------
    # Connect / disconnect datastock keys
    # ----------------------

    def connect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in self._dobj['canvas'].items():
            keyp = v0['handle'].mpl_connect('key_press_event', self.onkeypress)
            keyr = v0['handle'].mpl_connect('key_release_event', self.onkeypress)
            butp = v0['handle'].mpl_connect('button_press_event', self.mouseclic)
            res = v0['handle'].mpl_connect('resize_event', self.resize)
            butr = v0['handle'].mpl_connect('button_release_event', self.mouserelease)
            close = v0['handle'].mpl_connect('close_event', self.on_close)
            draw = v0['handle'].mpl_connect('draw_event', self.on_draw)
            # Make sure resizing is doen before resize_event
            # works without re-initializing because not a Qt Action
            v0['handle'].manager.toolbar.release = self.mouserelease
            # v0['handle'].manager.toolbar.release_zoom = self.mouserelease
            # v0['handle'].manager.toolbar.release_pan = self.mouserelease

            # make sure home button triggers background update
            # requires re-initializing because home is a Qt Action
            # only created by toolbar.addAction()
            v0['handle'].manager.toolbar.home = self.new_home
            try:
                # if _init_toolbar() implemented (matplotlib > )
                v0['handle'].manager.toolbar._init_toolbar()
            except NotImplementedError:
                v0['handle'].manager.toolbar.__init__(
                    v0['handle'],
                    v0['handle'].parent(),
                )
            except Exception as err:
                raise err

            self._dobj['canvas'][k0]['cid'] = {
                'keyp': keyp,
                'keyr': keyr,
                'butp': butp,
                'res': res,
                'butr': butr,
                'close': close,
            }

    def disconnect(self):
        if self._warn_ifnotInteractive():
            return
        for k0, v0 in self._dobj['canvas'].items():
            for k1, v1 in v0['cid'].items():
                v0['handle'].mpl_disconnect(v1)
            v0['handle'].manager.toolbar.release = lambda event: None

    # ----------------------
    # Connect / disconnect default keys
    # ----------------------

    def disconnect_old(self, force=False):

        if self._warn_ifnotInteractive():
            return

        if force:
            # disconnect key_press
            for k0, v0 in self._dobj['canvas'].items():
                v0['handle'].mpl_disconnect(
                    v0['handle'].manager.key_press_handler_id
                )
        else:
            lk = [kk for kk in list(plt.rcParams.keys()) if 'keymap' in kk]
            self.__store_rcParams = {}
            for kd in self._dobj['key'].keys():
                self.__store_rcParams[kd] = []
                for kk in lk:
                    if kd in plt.rcParams[kk]:
                        self.__store_rcParams[kd].append(kk)
                        plt.rcParams[kk].remove(kd)

        # disconnect button pick 
        for k0, v0 in self._dobj['canvas'].items():
            v0['handle'].mpl_disconnect(v0['handle'].button_pick_id)

    def reconnect_old(self):

        if self._warn_ifnotInteractive():
            return

        if self.__store_rcParams is not None:
            for kd in self.__store_rcParams.keys():
                for kk in self.__store_rcParams[kd]:
                    if kd not in plt.rcParams[kk]:
                        plt.rcParams[kk].append(kd)

    # ------------------------------------
    # Interactivity handling - preliminary
    # ------------------------------------

    def _get_current_grouprefdata_from_kax(self, kax=None):

        # Get current group and ref
        groupx = self._dobj['axes'][kax]['groupx']
        groupy = self._dobj['axes'][kax]['groupy']
        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']

        # Get kinter
        kinter = list(self._dobj['interactivity'].keys())[0]

        # Get current groups
        cur_groupx = self._dobj['interactivity'][kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][kinter]['cur_groupy']

        # determine whether cur_groupx shall be updated
        if groupx is not None:
            if cur_groupx in groupx:
                pass
            elif groupx is not None:
                cur_groupx = groupx[0]
        if groupy is not None:
            if cur_groupy in groupy:
                pass
            elif groupy is not None:
                cur_groupy = groupy[0]

        # # get current refs
        cur_refx = self._dobj['interactivity'][kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][kinter]['cur_refy']
        if groupx is not None:
            if cur_refx in self._dobj['group'][cur_groupx]['ref']:
                pass
            elif cur_groupx is not None:
                cur_refx = self._dobj['group'][cur_groupx]['ref'][0]
        if groupy is not None:
            if cur_refy in self._dobj['group'][cur_groupy]['ref']:
                pass
            elif cur_groupy is not None:
                cur_refy = self._dobj['group'][cur_groupy]['ref'][0]

        # data
        cur_datax = self._dobj['interactivity'][kinter]['cur_datax']
        if self._dobj['axes'][kax]['refx'] is not None:
            ix = self._dobj['axes'][kax]['refx'].index(cur_refx)
            cur_datax = self._dobj['axes'][kax]['datax'][ix]

        cur_datay = self._dobj['interactivity'][kinter]['cur_datay']
        if self._dobj['axes'][kax]['refy'] is not None:
            iy = self._dobj['axes'][kax]['refy'].index(cur_refy)
            cur_datay = self._dobj['axes'][kax]['datay'][iy]

        # Update interactivity dict
        self.kinter = kinter
        self._dobj['interactivity'][kinter].update({
            'cur_ax': kax,
            'cur_ax_panzoom': kax,
            'cur_groupx': cur_groupx,
            'cur_groupy': cur_groupy,
            'cur_refx': cur_refx,
            'cur_refy': cur_refy,
            'cur_datax': cur_datax,
            'cur_datay': cur_datay,
        })

    def _getset_current_axref(self, event):
        # Check click is relevant
        c0 = event.inaxes is not None and event.button == 1
        if not c0:
            raise Exception("clic not in axes")

        # get current ax key
        lkax = [
            k0 for k0, v0 in self._dobj['axes'].items()
            if v0['handle'] == event.inaxes
        ]
        kax = _generic_check._check_var(
            None, 'kax',
            types=str,
            allowed=lkax,
        )
        ax = self._dobj['axes'][kax]['handle']

        # Check axes is relevant and toolbar not active
        lc = [
            all([
                'fix' not in v0.keys()
                for v0 in self._dobj['axes'].values()
            ]),
            all([
                not v0['handle'].manager.toolbar.mode
                for v0 in self._dobj['canvas'].values()
            ]),
        ]
        if not all(lc):
            raise Exception("Not usable axes!")

        self._get_current_grouprefdata_from_kax(kax=kax)

    # -----------------------------
    # Interactivity: generic update
    # -----------------------------

    def update_interactivity(self):
        """ Called at each event """

        cur_groupx = self._dobj['interactivity'][self.kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][self.kinter]['cur_groupy']
        cur_refx = self._dobj['interactivity'][self.kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][self.kinter]['cur_refy']
        cur_datax = self._dobj['interactivity'][self.kinter]['cur_datax']
        cur_datay = self._dobj['interactivity'][self.kinter]['cur_datay']

        # Propagate indices through refs
        if cur_refx is not None:
            self.propagate_indices_per_ref(
                ref=cur_refx,
                lref=self._dobj['group'][cur_groupx]['ref'],
                ldata=self._dobj['group'][cur_groupx]['data'],
                param=None,
            )

        if cur_refy is not None:
            self.propagate_indices_per_ref(
                ref=cur_refy,
                lref=self._dobj['group'][cur_groupy]['ref'],
                ldata=self._dobj['group'][cur_groupy]['data'],
                param=None,
            )

        # get list of mobiles to update and set visible
        lmobiles = []
        if cur_groupx is not None:
            lmobiles += [
                k0 for k0, v0 in self._dobj['mobile'].items()
                if any([
                    rr in v0['ref']
                    for rr in self._dobj['group'][cur_groupx]['ref']
                ])
            ]

        if cur_groupy is not None:
            lmobiles += [
                k0 for k0, v0 in self._dobj['mobile'].items()
                if any([
                    rr in v0['ref']
                    for rr in self._dobj['group'][cur_groupy]['ref']
                ])
            ]

        self._update_mobiles(lmobiles=lmobiles) # 0.2 s

        if self.debug:
            self.show_debug()

    def _update_mobiles(self, lmobiles=None):

        # Set visibility of mobile objects - TBF/TBC
        for k0 in lmobiles:
            # all vs any ?
            vis = all([
                self._dobj['mobile'][k0]['ind']
                < self._dobj['group'][gg]['nmaxcur']
                for gg in self._dobj['mobile'][k0]['group_vis']
            ])
            self._dobj['mobile'][k0]['visible'] = vis

        # get list of axes to update
        lax = [
            k0 for k0, v0 in self._dobj['axes'].items()
            if any([self._dobj['mobile'][k1]['axes'] == k0 for k1 in lmobiles])
        ]

        # ---- Restore backgrounds ---- 1 ms
        for aa in lax:
            self._dobj['canvas'][
                self._dobj['axes'][aa]['canvas']
            ]['handle'].restore_region(
                self._dobj['axes'][aa]['bck'],
            )

        # ---- update data of group objects ----  0.15 s
        for k0 in lmobiles:
            _class2_interactivity._update_mobile(
                dmobile=self._dobj['mobile'],
                dref=self._dref,
                ddata=self._ddata,
                k0=k0,
            )

        # --- Redraw all objects (due to background restore) --- 25 ms
        for k0, v0 in self._dobj['mobile'].items():
            v0['handle'].set_visible(v0['visible'])
            try:
                self._dobj['axes'][v0['axes']]['handle'].draw_artist(v0['handle'])
            except Exception:
                print(0, k0)        # DB
                print(1, v0['axes'])    # DB
                print(2, self._dobj['axes'][v0['axes']]['handle'])  # DB
                print(3, v0['handle'])  # DB

        # ---- blit axes ------ 5 ms
        for aa in lax:
            self._dobj['canvas'][
                self._dobj['axes'][aa]['canvas']
            ]['handle'].blit(self._dobj['axes'][aa]['handle'].bbox)

    # ----------------------
    # Interactivity: resize
    # ----------------------

    def resize(self, event):
        _class2_interactivity._set_dbck(
            lax=self._dobj['axes'].keys(),
            daxes=self._dobj['axes'],
            dcanvas=self._dobj['canvas'],
            dmobile=self._dobj['mobile'],
            event=event,
        )

    def new_home(self, *args):
        for k0, v0 in self._dobj['canvas'].items():
            super(
                v0['handle'].manager.toolbar.__class__,
                v0['handle'].manager.toolbar,
            ).home(*args)
        _class2_interactivity._set_dbck(
            lax=self._dobj['axes'].keys(),
            daxes=self._dobj['axes'],
            dcanvas=self._dobj['canvas'],
            dmobile=self._dobj['mobile'],
            event=None,
        )

    def on_draw(self, event):
        pass

    # ----------------------
    # Interactivity: mouse
    # ----------------------

    def mouseclic(self, event):

        # Check click is relevant
        c0 = event.button == 1
        if not c0:
            return

        # get / set cuurent interactive usabel axes and ref
        try:
            self._getset_current_axref(event)
        except Exception as err:
            if str(err) in ['clic not in axes', "Not usable axes!"]:
                return
            raise err
            # warnings.warn(str(err))
            # return

        kinter = self.kinter
        kax = self._dobj['interactivity'][kinter]['cur_ax']
        ax = self._dobj['axes'][kax]['handle']

        refx = self._dobj['axes'][kax]['refx']
        refy = self._dobj['axes'][kax]['refy']
        if refx is None and refy is None:
            return

        cur_groupx = self._dobj['interactivity'][kinter]['cur_groupx']
        cur_groupy = self._dobj['interactivity'][kinter]['cur_groupy']
        cur_refx = self._dobj['interactivity'][kinter]['cur_refx']
        cur_refy = self._dobj['interactivity'][kinter]['cur_refy']
        cur_datax = self._dobj['interactivity'][kinter]['cur_datax']
        cur_datay = self._dobj['interactivity'][kinter]['cur_datay']

        shift = self._dobj['key']['shift']['val']
        ctrl = any([self._dobj['key'][ss]['val'] for ss in ['control', 'ctrl']])

        # Update number of indices (for visibility)
        gax = []
        if self._dobj['axes'][kax]['groupx'] is not None:
            gax += self._dobj['axes'][kax]['groupx']
        if self._dobj['axes'][kax]['groupy'] is not None:
            gax += self._dobj['axes'][kax]['groupy']
        for gg in set([cur_groupx, cur_groupy]):
            if gg is not None and gg in gax:
                out = _class2_interactivity._update_indices_nb(
                    group=gg,
                    dgroup=self._dobj['group'],
                    ctrl=ctrl,
                    shift=shift,
                )
                if out is False:
                    return

        # update ref indices
        if None not in [cur_refx, cur_refy] and cur_refx == cur_refy:

            cdx = self._ddata[cur_datax]['data']
            cdy = self._ddata[cur_datay]['data']
            dx = np.diff(ax.get_xlim())
            dy = np.diff(ax.get_ylim())
            dist = ((cdx - event.xdata)/dx)**2 + ((cdy - event.ydata)/dy)**2
            if dist.ndim == 1:
                ix = np.nanargmin(dist)
            elif dist.ndim == 2:
                axis = self._ddata[cur_datax]['ref'].index(cur_refx)
                ix = np.nanargmin(np.nanmin(dist, axis=1-axis))
            else:
                raise NotImplementedError()
            c0x = True
            c0y = False

        else:

            c0x = (
                cur_refx is not None
                and self._dobj['axes'][kax]['refx'] is not None
                and cur_refx in self._dobj['axes'][kax]['refx']
            )
            if c0x:
                monot = None
                c0 = (
                    cur_datax == 'index'
                    or self._ddata[cur_datax]['data'].dtype.type == np.str_
                )
                if c0:
                    cdx = 'index'
                else:
                    monot = self._ddata[cur_datax]['monot'] == (True,)
                    cdx = self._ddata[cur_datax]['data']
                ix = _class1_compute._get_index_from_data(
                    data=cdx,
                    data_pick=np.r_[event.xdata],
                    monot=monot,
                )[0]

            c0y = (
                cur_refy is not None
                and self._dobj['axes'][kax]['refy'] is not None
                and cur_refy in self._dobj['axes'][kax]['refy']
            )
            if c0y:
                monot = None
                c0 = (
                    cur_datay == 'index'
                    or self._ddata[cur_datay]['data'].dtype.type == np.str_
                )
                if c0:
                    cdy = 'index'
                else:
                    monot = self._ddata[cur_datay]['monot'] == (True,)
                    cdy = self._ddata[cur_datay]['data']
                iy = _class1_compute._get_index_from_data(
                    data=cdy,
                    data_pick=np.r_[event.ydata],
                    monot=monot,
                )[0]

        # Update ref indices
        if c0x:
            cur_ix = self._dobj['group'][cur_groupx]['indcur']
            follow = (
                cur_ix == self._dobj['group'][cur_groupx]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[cur_refx]['indices'][cur_ix:] = ix
            else:
                self._dref[cur_refx]['indices'][cur_ix] = ix

        # Update ref indices
        if c0y:
            cur_iy = self._dobj['group'][cur_groupy]['indcur']
            follow = (
                cur_iy == self._dobj['group'][cur_groupy]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[cur_refy]['indices'][cur_iy:] = iy
            else:
                self._dref[cur_refy]['indices'][cur_iy] = iy

        self.update_interactivity()

    def mouserelease(self, event):
        """ Mouse release: nothing except if resize ongoing (redraw bck) """

        c0 = event.inaxes is not None and event.button == 1
        if not c0:
            return

        can = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if v0['handle'] == event.inaxes.figure.canvas
        ][0]
        mode = self._dobj['canvas'][can]['handle'].manager.toolbar.mode.lower()
        c0 = 'pan' in  mode
        c1 = 'zoom' in mode

        if c0 or c1:
            kax = self._dobj['interactivity'][self.kinter]['cur_ax_panzoom']
            if kax is None:
                msg = (
                    "Make sure you release the mouse button on an axes !"
                    "\n Otherwise the background plot cannot be properly updated !"
                )
                raise Exception(msg)
            ax = self._dobj['axes'][kax]['handle']
            lax = ax.get_shared_x_axes().get_siblings(ax)
            lax += ax.get_shared_y_axes().get_siblings(ax)
            lax = list(set(lax))
            lax = [
                [
                    kax for kax in self._dobj['axes'].keys()
                    if self._dobj['axes'][kax]['handle'] == ax
                ][0]
                for ax in lax
            ]
            _class2_interactivity._set_dbck(
                lax=lax,
                daxes=self._dobj['axes'],
                dcanvas=self._dobj['canvas'],
                dmobile=self._dobj['mobile'],
            )

    # ----------------------
    # Interactivity: keys
    # ----------------------

    def onkeypress(self, event):
        """ Event handler in case of key press / release """

        # -----------------------
        # Check event is relevant 1

        # decompose key combinations
        lkey = event.key.split('+')

        # get current inter, axes, canvas
        kinter = self.kinter
        kax = self._dobj['interactivity'][kinter]['cur_ax']
        kcan = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if k0 == self._dobj['axes'][kax]['canvas']
        ][0]
        can = self._dobj['canvas'][kcan]['handle']

        # check relevance
        c0 = can.manager.toolbar.mode != ''
        c1 = len(lkey) not in [1, 2]
        c2 = [ss not in self._dobj['key'].keys() for ss in lkey]

        if c0 or c1 or any(c2):
            return

        # -----------------------
        # Check event is relevant 2

        # get list of current keys for each action type
        lgen = [kk for kk in self.__dkeys_r['generic'] if kk in lkey]
        lmov = [kk for kk in self.__dkeys_r['move'] if kk in lkey]
        lgrp = [kk for kk in self.__dkeys_r['group'] if kk in lkey]
        lind = [kk for kk in self.__dkeys_r['indices'] if kk in lkey]

        # if no relevant key pressed => return
        ngen, nmov, ngrp, nind = len(lgen), len(lmov), len(lgrp), len(lind)
        ln = np.r_[ngen, nmov, ngrp, nind]
        if np.any(ln > 1) or np.sum(ln) > 2:
            return
        if np.sum(ln) == 2 and (ngrp == 1 or nind ==1 ):
            return

        # only keep relevant keys
        genk = None if ngen == 0 else lgen[0]
        movk = None if nmov == 0 else lmov[0]
        grpk = None if ngrp == 0 else lgrp[0]
        indk = None if nind == 0 else lind[0]

        # ------------------------
        # Event = change key value

        # change key values if relevant
        if event.name == 'key_release_event':
            if event.key == genk:
                self._dobj['key'][genk]['val'] = False
            return

        if genk is not None and event.key == genk:
            self._dobj['key'][genk]['val'] = True
            return

        # ----------------------------
        # Event = change current group

        if grpk is not None:
            # group
            group = self._dobj['key'][event.key]['group']
            cx = any([
                v0['groupx'] is not None and  group in v0['groupx']
                for v0 in self._dobj['axes'].values()
            ])
            if cx:
                self._dobj['interactivity'][self.kinter]['cur_groupx'] = group
            cy = any([
                v0['groupy'] is not None and group in v0['groupy']
                for v0 in self._dobj['axes'].values()
            ])
            if cy:
                self._dobj['interactivity'][self.kinter]['cur_groupy'] = group

            # axes
            cur_ax = self._dobj['interactivity'][self.kinter]['cur_ax']
            if cur_ax not in self._dobj['group'][group]['axes']:
                self._dobj['interactivity'][self.kinter]['cur_ax'] = (
                    self._dobj['group'][group]['axes'][0]
                )

            # ref
            if cx:
                cur_refx = self._dobj['interactivity'][self.kinter]['cur_refx']
                if self._dref[cur_refx]['group'] != group:
                    cur_refx = self._dobj['group'][group]['ref'][0]
                self._dobj['interactivity'][self.kinter]['cur_refx'] = cur_refx

            if cy:
                cur_refy = self._dobj['interactivity'][self.kinter]['cur_refy']
                if self._dref[cur_refy]['group'] != group:
                    cur_refy = self._dobj['group'][group]['ref'][0]
                self._dobj['interactivity'][self.kinter]['cur_refy'] = cur_refy

            # data
            if c0:
                self._dobj['interactivity'][self.kinter]['cur_datax'] = 'index'
            if c0:
                self._dobj['interactivity'][self.kinter]['cur_datay'] = 'index'

            msg = f"Current group set to {group}"
            print(msg)
            return

        # ----------------------------
        # Event = change current index

        if indk is not None:
            groupx = self._dobj['interactivity'][self.kinter]['cur_groupx']
            groupy = self._dobj['interactivity'][self.kinter]['cur_groupy']

            # groupx
            if groupx is not None:
                imax = self._dobj['group'][groupx]['nmaxcur']
                ii = int(event.key)
                if ii > imax:
                    msg = "Set to current max index for group '{groupx}': {imax}"
                    print(msg)
                ii = min(ii, imax)
                self._dobj['group'][groupx]['indcur'] = ii

            # groupy
            if groupy is not None:
                imax = self._dobj['group'][groupy]['nmaxcur']
                ii = int(event.key)
                if ii > imax:
                    msg = "Set to current max index for group '{groupy}': {imax}"
                    print(msg)
                ii = min(ii, imax)
                self._dobj['group'][groupy]['indcur'] = ii

            msg = f"Current indices set to {ii}"
            print(msg)
            return

        # ----------------------------
        # Event = move current index

        if movk is not None:

            if movk in ['left', 'right']:
                group = self._dobj['interactivity'][self.kinter]['cur_groupx']
                ref = self._dobj['interactivity'][self.kinter]['cur_refx']
                incsign = 1.
                if self._dobj['axes'][kax].get('invertx', False):
                    incsign = -1
            elif movk in ['up', 'down']:
                group = self._dobj['interactivity'][self.kinter]['cur_groupy']
                ref = self._dobj['interactivity'][self.kinter]['cur_refy']
                incsign = 1.
                if self._dobj['axes'][kax].get('inverty', False):
                    incsign = -1

            if group is None:
                return

            # dmovkeys for inversions and steps ?

            shift = self._dobj['key']['shift']['val']
            ctrl = any([
                self._dobj['key'][ss]['val'] for ss in ['control', 'ctrl']
            ])
            alt = self._dobj['key']['alt']['val']

            # Check max number of occurences not reached if shift
            c0 = (
                shift
                and (
                    self._dobj['group'][group]['indcur']
                    == self._dobj['group'][group]['nmax'] - 1
                )
            )
            if c0:
                msg = "Max nb. of plots reached ({0}) for group {1}"
                msg = msg.format(self._dobj['group'][group]['nmax'], group)
                print(msg)
                return

            # update nb of visible indices
            out = _class2_interactivity._update_indices_nb(
                group=group,
                dgroup=self._dobj['group'],
                ctrl=ctrl,
                shift=shift,
            )
            if out is False:
                return

            # get increment from key
            cax = self._dobj['interactivity'][self.kinter]['cur_ax']
            inc = incsign * (
                self._dref[ref]['inc'][int(alt)]
                * self._dobj['axes'][cax]['inc'][movk]
            )

            # update ref indices
            icur = self._dobj['group'][group]['indcur']
            ix = (
                (self._dref[ref]['indices'][icur] + inc)
                % self._dref[ref]['size']
            )

            # Update ref indices
            follow = (
                icur == self._dobj['group'][group]['nmaxcur'] - 1
                and self._dobj['interactivity'][kinter]['follow']
            )
            if follow:
                self._dref[ref]['indices'][icur:] = ix
            else:
                self._dref[ref]['indices'][icur] = ix

            # global update of interactivity
            self.update_interactivity()

    # -------------------
    # Close all
    # -------------------

    def on_close(self, event):
        kcan = [
            k0 for k0, v0 in self._dobj['canvas'].items()
            if v0['handle'] == event.canvas
        ]
        if len(kcan) > 1:
            raise Exception('Several matching canvas')
        elif len(kcan) == 1:

            if len(self._dobj['canvas']) == 1:
                self.close_all()

            else:
                lax = [
                    k1 for k1, v1 in self._dobj['axes'].items()
                    if v1['canvas'] == kcan[0]
                ]
                lmob = [
                    k1 for k1, v1 in self._dobj['mobile'].items()
                    if v1['axes'] in lax
                ]
                for k1 in lax:
                    del self._dobj['axes'][k1]
                for k1 in lmob:
                    del self._dobj['mobile'][k1]
                del self._dobj['canvas'][kcan[0]]

    def close_all(self):

        # close figures
        if 'axes' in self._dobj.keys():
            lfig = set([
                v0['handle'].figure for v0 in self._dobj['axes'].values()
            ])
            for ff in lfig:
                plt.close(ff)

        # delete obj dict
        lk = ['interactivity', 'mobile', 'key', 'canvas', 'group', 'axes']
        for kk in lk:
            if kk in self._dobj.keys():
                del self._dobj[kk]

        # remove interactivity-specific param in dref
        lp = list(set(self.get_lparam(which='ref')).intersection(
            ['indices', 'group', 'inc']
        ))
        self.remove_param(which='ref', param=lp)


# #############################################################################
# #############################################################################
#            set __all__
# #############################################################################


__all__ = [
    sorted([k0 for k0 in locals() if k0.startswith('DataStock')])[-1]
]
