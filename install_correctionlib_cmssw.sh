# no shebang, must be sourced

# based on install script by Pieter David:
# https://gist.github.com/pieterdavid/8f43f302e9f8a71f92702101600b7ddb
# https://gist.githubusercontent.com/pieterdavid/8f43f302e9f8a71f92702101600b7ddb/raw/84be84be0bf7b11336cfee2d068da686fd130205/install_correctionlib.sh

# run via: source ./install_correctionlib.sh

install_correctionlib_cmssw() {

    local pycommand="python3"  # force use of python 3
    local pyvers="py3"
    local correctionlib_version="2.5.0"
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local pip_tmpdir="${this_dir}/.python"
    local orig="${PWD}"

    # fail if not inside CMSSW work area
    if [ -z "$CMSSW_BASE" ]; then
        echo "[ERROR] You must use this package inside a CMSSW environment."
        return 1
    fi

    # check if correctionlib tool is already installed
    local toolname="${pyvers}-correctionlib-new"
    scram tool info "${toolname}" > /dev/null 2> /dev/null
    if [ $? -eq 0 ]; then
        echo "[INFO] CMSSW tool '${toolname}' already installed; nothing to do."
        return 0
    fi

    local cmssw_tool_install_dir="${CMSSW_BASE}/install/${toolname}"
    if [ -d "${cmssw_tool_install_dir}" ]; then
        echo "[ERROR] Install path ${cmssw_tool_install_dir} already exists; please remove by hand and try again if you want to reinstall."
        return 1
    fi

    local pymajmin=$(${pycommand} -c 'import sys; print(".".join(str(num) for num in sys.version_info[:2]))')
    echo "[INFO] Installing as ${toolname} with python=${pycommand} (${pymajmin}) to: ${cmssw_tool_install_dir}"

    # back up original values of environment variables
    local bk_pythonpath="${PYTHONPATH}"
    local bk_path="${PATH}"
    local bk_tmpdir="${TMPDIR}"

    # use pip to download correctionlib package
    export TMPDIR="${CMSSW_BASE}/tmp"
    ( ${pycommand} -m pip --version && ${pycommand} -m pip download -v correctionlib==${correctionlib_version} -d "${TMPDIR}" ) >/dev/null 2>/dev/null

    # on failure, boostrap newer version of pip
    if [ $? -ne 0 ]; then
        echo "[INFO] No working pip found, bootstrapping in: ${pip_tmpdir}"
        [ -d "${pip_tmpdir}" ] || mkdir "${pip_tmpdir}"
        if [ ! -f "${pip_tmpdir}/bin/pip" ]; then
            wget -q -O "${pip_tmpdir}/get-pip.py" "https://bootstrap.pypa.io/pip/${pymajmin}/get-pip.py"
            ${pycommand} "${pip_tmpdir}/get-pip.py" --prefix="${pip_tmpdir}" --ignore-installed
        fi
        export PYTHONPATH="${pip_tmpdir}/lib/python${pymajmin}/site-packages:${PYTHONPATH}"
        export PATH="${pip_tmpdir}/bin:${PATH}"
        ${pycommand} -m pip install --prefix="${pip_tmpdir}" --ignore-installed setuptools_scm scikit-build 'cmake>=3.11'
    fi

    # install correctionlib package via pip
    echo "[INFO] Installing correctionlib."
    mkdir -p ${cmssw_tool_install_dir}
    ${pycommand} -m pip install --prefix="${cmssw_tool_install_dir}" --no-binary=correctionlib==${correctionlib_version} --ignore-installed correctionlib
    local correctionlib_version=$(PYTHONPATH="${cmssw_tool_install_dir}/lib/python${pymajmin}/site-packages" ${pycommand} -m pip show correctionlib | grep Version | sed 's/Version: //')

    # remove temporary pip directory
    if [ -d "${pip_tmpdir}" ]; then
        rm -rf "${pip_tmpdir}"
    fi

    # restore original values of environment variables
    export PYTHONPATH="${bk_pythonpath}"
    export PATH="${bk_path}"
    export TMPDIR="${bk_tmpdir}"

    local pyversu=$(echo "${pyvers}" | tr 'a-z' 'A-Z')

    # write XML file for CMSSW tool configuration
    local toolfile="${cmssw_tool_install_dir}/${toolname}.xml"
    echo "[INFO] Writing tool config to: ${toolfile}"
    cat <<EOF_TOOLFILE >"${toolfile}"
<tool name="${toolname}" version="${correctionlib_version}">
  <info url="https://github.com/cms-nanoAOD/correctionlib"/>
  <client>
    <environment name="${pyversu}_CORRECTIONLIB_BASE" default="${cmssw_tool_install_dir}"/>
    <runtime name="LD_LIBRARY_PATH"     value="\$${pyversu}_CORRECTIONLIB_BASE/lib" type="path"/>
    <runtime name="PYTHONPATH"          value="\$${pyversu}_CORRECTIONLIB_BASE/lib/python${pymajmin}/site-packages" type="path"/>
EOF_TOOLFILE
    if [[ "${pyvers}" == "py3" ]]; then
      cat <<EOF_TOOLFILE >>"${toolfile}"
    <runtime name="PATH"                value="\$${pyversu}_CORRECTIONLIB_BASE/bin" type="path"/>
EOF_TOOLFILE
    fi
    cat <<EOF_TOOLFILE >>"${toolfile}"
  </client>
</tool>
EOF_TOOLFILE

    echo "[INFO] Updating environment"
    scram setup "${toolfile}"
    eval `scramv1 runtime -sh`  # cmsenv

    echo "[INFO] Successfully installed tool '${toolname}'."
}

install_correctionlib_cmssw "$@"
