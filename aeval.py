# https://github.com/penn5/meval/blob/master/meval/__init__.py

import ast


async def aeval(code: str, globs: dict, **kwargs) -> any:
    _locs = {}
    globs = globs.copy()

    _globs = "_globs"
    while _globs in globs:
        _globs = "_" + _globs

    kwargs[_globs] = {k: globs[k] for k in ["__name__", "__package__"]}

    root = ast.parse(code, "exec")
    code = root.body

    _ret = "_ret"
    while _ret in globs or any(
        isinstance(k, ast.Name) and k.id == _ret for k in ast.walk(root)
    ):
        _ret = "_" + _ret

    if not code:
        return None

    if not any(isinstance(k, ast.Return) for k in code):
        for i, k in enumerate(code):
            if isinstance(k, ast.Expr) and (
                i == len(code) - 1 or not isinstance(k.value, ast.Call)
            ):
                code[i] = ast.Expr(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=_ret, ctx=ast.Load()),
                            attr="append",
                            ctx=ast.Load(),
                        ),
                        args=[k.value],
                        keywords=[],
                    )
                )
    else:
        for k in code:
            if isinstance(k, ast.Return):
                k.value = ast.List(elts=[k.value], ctx=ast.Load())

    code.append(ast.Return(value=ast.Name(id=_ret, ctx=ast.Load())))

    glob_copy = ast.Expr(
        ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="globals", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                attr="update",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(
                    arg=None,
                    value=ast.Name(id=_globs, ctx=ast.Load()),
                )
            ],
        )
    )
    ast.fix_missing_locations(glob_copy)
    code.insert(0, glob_copy)

    sign = ast.Assign(
        targets=[ast.Name(id=_ret, ctx=ast.Store())],
        value=ast.List(elts=[], ctx=ast.Load()),
    )
    ast.fix_missing_locations(sign)
    code.insert(1, sign)

    args = ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[ast.arg(arg=arg, annotation=None) for arg in kwargs],
        kwarg=None,
        defaults=[],
        kw_defaults=[None] * len(kwargs),
    )
    func = ast.AsyncFunctionDef(
        name="_", args=args, body=code, decorator_list=[], returns=None
    )
    ast.fix_missing_locations(func)

    _mod = ast.Module(body=[func], type_ignores=[])
    comp = compile(_mod, "<string>", "exec")
    exec(comp, {}, _locs)

    res = await _locs["_"](**kwargs)
    res = [await i if hasattr(i, "__await__") else i for i in res]
    res = [i for i in res if i is not None]

    return res[0] if len(res) == 1 else res if res else None
